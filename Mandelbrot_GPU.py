# --- Imports ---
import pygame
import numpy as np
import matplotlib
import matplotlib.colors as colors
from concurrent.futures import ThreadPoolExecutor
# Import cuda and math for kernel, prange for parallel CPU
from numba import jit, NumbaWarning, cuda, prange, float32, uint32
import time
import sys
import traceback
import warnings
import math # Standard math for host code

# --- Configuration ---
WIDTH, HEIGHT = 800, 600
INITIAL_MAX_ITER = 150
ADAPTIVE_ITER = True
COLORMAP_NAME = 'plasma'
THREADS_PER_BLOCK = (16, 16)
# Data type for smooth iteration counts
ITER_DTYPE = np.float32

# warnings.simplefilter('ignore', category=NumbaWarning)

# --- CUDA Mandelbrot Kernel (Smooth Coloring) ---
@cuda.jit(device=True)
def mandel_point_smooth_gpu(creal, cimag, max_iter):
    """Calculate smooth iterations for a single point (GPU)."""
    # Use float64 for intermediate calculations for precision
    z_real = 0.0
    z_imag = 0.0
    max_iter_f = float32(max_iter) # Float version of max_iter

    # Interior Checking (scalar version) - return max_iter if inside
    cr_minus_1_4 = creal - 0.25
    q = cr_minus_1_4 * cr_minus_1_4 + cimag * cimag
    if q * (q + cr_minus_1_4) < 0.25 * cimag * cimag:
        return max_iter_f
    cr_plus_1 = creal + 1.0
    if cr_plus_1 * cr_plus_1 + cimag * cimag < 0.0625:
        return max_iter_f

    for n in range(max_iter):
        z_real_sq = z_real * z_real
        z_imag_sq = z_imag * z_imag
        mag_sq = z_real_sq + z_imag_sq

        if mag_sq > 4.0:
            # Escaped - calculate smooth value
            # Use sqrt from 'math' module (supported in Numba CUDA device code)
            abs_z = math.sqrt(mag_sq)
            # Formula: n + 1 - log(log(|z|))/log(2)
            # Add small epsilon to prevent log(log(1)) issues if |z| is exactly 1 (unlikely here)
            smooth = float32(n + 1.0 - math.log(math.log(abs_z + 1e-10)) / math.log(2.0))
            # Clamp value just below max_iter to avoid confusion with non-escaped points
            return min(smooth, max_iter_f - 0.001) # Return smooth value as float32

        # Iterate
        z_imag = 2.0 * z_real * z_imag + cimag
        z_real = z_real_sq - z_imag_sq + creal

    # Did not escape
    return max_iter_f # Return float max_iter

@cuda.jit
def mandel_kernel_smooth(c_real_g, c_imag_g, max_iter, iterations_out_g):
    """CUDA Kernel to compute smooth Mandelbrot iterations."""
    y, x = cuda.grid(2)
    height, width = iterations_out_g.shape
    if y < height and x < width:
        creal = c_real_g[y, x]
        cimag = c_imag_g[y, x]
        # Store result (which is float32)
        iterations_out_g[y, x] = mandel_point_smooth_gpu(creal, cimag, max_iter)


# --- CPU Fallback Calculation Function (Parallel Per-Pixel, Smooth Coloring) ---
# Note: We rewrite this to be per-pixel for smooth coloring and parallel=True
@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_mandelbrot_iterations_cpu_smooth(c_real, c_imag, max_iter):
    """CPU version using Numba JIT, parallel=True, per-pixel loop, smooth coloring."""
    height, width = c_real.shape
    # Output array is now float32
    iterations_out = np.empty((height, width), dtype=ITER_DTYPE)
    max_iter_f = float32(max_iter)

    # Use prange for parallel loop execution over rows
    for y in prange(height):
        for x in range(width):
            creal = c_real[y, x]
            cimag = c_imag[y, x]

            # --- Scalar calculation copied from mandel_point_smooth_gpu ---
            # Interior Checking
            cr_minus_1_4 = creal - 0.25
            q = cr_minus_1_4 * cr_minus_1_4 + cimag * cimag
            is_inside = False
            if q * (q + cr_minus_1_4) < 0.25 * cimag * cimag:
                is_inside = True
            else:
                cr_plus_1 = creal + 1.0
                if cr_plus_1 * cr_plus_1 + cimag * cimag < 0.0625:
                    is_inside = True

            if is_inside:
                iterations_out[y, x] = max_iter_f
                continue # Skip iteration loop for this pixel

            # Iteration
            z_real = 0.0
            z_imag = 0.0
            escaped = False
            smooth_val = max_iter_f
            for n in range(max_iter):
                z_real_sq = z_real * z_real
                z_imag_sq = z_imag * z_imag
                mag_sq = z_real_sq + z_imag_sq

                if mag_sq > 4.0:
                    # Escaped - calculate smooth value
                    abs_z = math.sqrt(mag_sq)
                    # log/log might fail if abs_z is very close to 1, add epsilon
                    smooth = float32(n + 1.0 - math.log(math.log(abs_z + 1e-10)) / math.log(2.0))
                    smooth_val = min(smooth, max_iter_f - 0.001)
                    escaped = True
                    break # Exit loop once escaped

                z_imag = 2.0 * z_real * z_imag + cimag
                z_real = z_real_sq - z_imag_sq + creal

            iterations_out[y, x] = smooth_val # Assign final value (smooth or max_iter_f)
            # --- End scalar calculation ---

    return iterations_out


# --- Color Mapping Function Generator (Handles Float Input) ---
def get_colormap_func(cmap_name):
    """Returns a function that applies colormap (handles float iterations)."""
    try: cmap = matplotlib.colormaps[cmap_name]
    except AttributeError:
        import matplotlib.cm as cm
        print("Warning: Using deprecated cm.get_cmap(). Update Matplotlib if possible.")
        cmap = cm.get_cmap(cmap_name)

    def apply_colormap(iterations, current_max_iter): # iterations is now float
        norm = colors.Normalize(vmin=0, vmax=current_max_iter) # Normalize float values
        max_iter_f_cmp = float32(current_max_iter)

        # Identify points that reached max_iter (compare floats carefully, or check >=)
        # Using >= ensures points clamped just below max_iter also get black.
        inside_mask = iterations >= max_iter_f_cmp

        rgba_colors = cmap(norm(iterations))
        rgb_colors = (rgba_colors[:, :, :3] * 255).astype(np.uint8)

        # Boolean indexing works fine outside Numba here
        rgb_colors[inside_mask] = [0, 0, 0]
        return np.ascontiguousarray(rgb_colors)

    return apply_colormap


# --- Main Application Class (Adapts for Smooth Coloring) ---
class MandelbrotExplorer:
    def __init__(self, width, height, initial_max_iter, colormap_name):
        pygame.init()
        self.use_gpu = False
        try:
            if cuda.is_available():
                print("CUDA device found. Attempting to use GPU.")
                cuda.detect(); cuda.synchronize()
                self.use_gpu = True; print("CUDA initialized successfully.")
            else: print("No CUDA device found. Using CPU.")
        except Exception as e:
            print(f"Error checking/initializing CUDA: {e}. Falling back to CPU.")
            self.use_gpu = False

        caption = "Interactive Mandelbrot Explorer " + ("(CUDA Smooth)" if self.use_gpu else "(CPU Parallel Smooth)")
        pygame.display.set_caption(caption)

        self.width = width; self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self.font = pygame.font.SysFont(None, 24); self.clock = pygame.time.Clock()
        self.center_x = -0.7; self.center_y = 0.0; self.zoom = 1.0
        self.view_width = 4.0 / self.zoom
        self.max_iter = initial_max_iter
        self.colormap_name = colormap_name
        self.mandel_surface = pygame.Surface((self.width, self.height)); self.mandel_surface.fill((0,0,0))
        self.apply_colormap = get_colormap_func(self.colormap_name)
        self.panning = False; self.pan_start_pos = None; self.pan_start_center = None
        self.executor = ThreadPoolExecutor(max_workers=1) # Single thread for task submission
        self.calculation_future = None; self.needs_update = True
        self.last_calc_params = None; self.current_render_id = 0; self.next_render_id = 0

    # _get_complex_coords remains the same
    def _get_complex_coords(self, width, height, cx, cy, zoom):
        view_width = 4.0 / zoom; scale = view_width / width
        real_min=cx-(width/2)*scale; real_max=cx+(width/2)*scale
        imag_min=cy-(height/2)*scale; imag_max=cy+(height/2)*scale
        c_real_1d = np.linspace(real_min, real_max, width, dtype=np.float64)
        c_imag_1d = np.linspace(imag_max, imag_min, height, dtype=np.float64)
        c_real, c_imag = np.meshgrid(c_real_1d, c_imag_1d)
        return np.ascontiguousarray(c_real), np.ascontiguousarray(c_imag)

    # _pixel_to_complex remains the same
    def _pixel_to_complex(self, px, py):
        scale = self.view_width / self.width
        real = self.center_x + (px - self.width / 2) * scale
        imag = self.center_y + (self.height / 2 - py) * scale
        return complex(real, imag)

    # --- Calculation Tasks ---
    @staticmethod
    def _gpu_calculation_task(cx, cy, zoom, width, height, max_iter, render_id):
        """Task for GPU calculation (smooth)."""
        d_c_real, d_c_imag, d_iterations = None, None, None # Ensure cleanup scope
        try:
            start_time = time.time()
            print(f"Starting GPU calculation ID {render_id} ({width}x{height}, iter={max_iter})...")
            host_c_real, host_c_imag = MandelbrotExplorer._get_complex_coords(None, width, height, cx, cy, zoom)

            # Allocate GPU memory (use ITER_DTYPE for output)
            d_c_real = cuda.to_device(host_c_real)
            d_c_imag = cuda.to_device(host_c_imag)
            d_iterations = cuda.device_array((height, width), dtype=ITER_DTYPE) # Use float

            threads_pb = THREADS_PER_BLOCK
            blocks_x = math.ceil(width / threads_pb[1])
            blocks_y = math.ceil(height / threads_pb[0])
            blocks_pg = (blocks_y, blocks_x)

            # Launch SMOOTH kernel
            mandel_kernel_smooth[blocks_pg, threads_pb](d_c_real, d_c_imag, max_iter, d_iterations)
            cuda.synchronize() # Wait for GPU

            host_iterations = d_iterations.copy_to_host() # Get float results
            end_time = time.time(); calc_time = end_time - start_time
            print(f"GPU Calculation ID {render_id} finished in {calc_time:.3f}s")

            del d_c_real, d_c_imag, d_iterations # Explicit cleanup
            return { 'iterations': host_iterations, 'render_id': render_id, 'calc_time': calc_time,
                     'params': {'cx': cx, 'cy': cy, 'zoom': zoom, 'width': width, 'height': height, 'max_iter': max_iter} }
        except Exception as e:
            print(f"Error in GPU calculation (ID {render_id}): {e}"); traceback.print_exc()
            try: del d_c_real; del d_c_imag; del d_iterations
            except: pass
            return {'error': True, 'render_id': render_id}

    @staticmethod
    def _cpu_calculation_task(cx, cy, zoom, width, height, max_iter, render_id):
        """Task for CPU calculation (smooth, parallel)."""
        try:
            start_time = time.time()
            print(f"Starting CPU calculation ID {render_id} ({width}x{height}, iter={max_iter})...")
            host_c_real, host_c_imag = MandelbrotExplorer._get_complex_coords(None, width, height, cx, cy, zoom)
            # Call the new parallel, smooth CPU function
            host_iterations = compute_mandelbrot_iterations_cpu_smooth(host_c_real, host_c_imag, max_iter)
            end_time = time.time(); calc_time = end_time - start_time
            print(f"CPU Calculation ID {render_id} finished in {calc_time:.3f}s")
            return { 'iterations': host_iterations, 'render_id': render_id, 'calc_time': calc_time,
                     'params': {'cx': cx, 'cy': cy, 'zoom': zoom, 'width': width, 'height': height, 'max_iter': max_iter} }
        except Exception as e:
            print(f"Error in CPU calculation (ID {render_id}): {e}"); traceback.print_exc()
            return {'error': True, 'render_id': render_id}

    # Callback remains the same (already handles 'iterations' key)
    def _calculation_done_callback(self, future):
        try:
            result = future.result()
            if result.get('error', False):
                 print(f"Calculation task ID {result.get('render_id', 'unknown')} failed.")
                 return
            if result['render_id'] == self.current_render_id:
                params = result['params']
                if params['width'] == self.width and params['height'] == self.height:
                    print(f"Received iteration data for ID {result['render_id']}")
                    iterations = result['iterations'] # Now float32
                    try:
                        rgb_array = self.apply_colormap(iterations, params['max_iter'])
                    except Exception as e:
                         print(f"Error applying colormap: {e}"); traceback.print_exc(); return
                    surface = pygame.surfarray.make_surface(np.transpose(rgb_array, (1, 0, 2)))
                    self.mandel_surface = surface; self.last_calc_params = params
                else:
                    print(f"Warning: Data size != window size. Requesting update.")
                    self._request_update()
            else:
                print(f"Ignoring stale calculation result (ID {result['render_id']}, expected {self.current_render_id})")
        except Exception as e:
            print(f"Error processing calculation result: {e}"); traceback.print_exc()

    # _request_update remains the same
    def _request_update(self):
        self.needs_update = True; self.next_render_id += 1

    # _start_calculation remains the same (chooses CPU/GPU task)
    def _start_calculation(self):
        if self.needs_update:
            current_params = { 'cx': self.center_x, 'cy': self.center_y, 'zoom': self.zoom,
                               'width': self.width, 'height': self.height, 'max_iter': self.max_iter }
            if current_params != self.last_calc_params:
                self.needs_update = False; self.current_render_id = self.next_render_id
                backend = 'GPU' if self.use_gpu else 'CPU'
                print(f"Submitting calculation ID {self.current_render_id} ({backend}) with max_iter={self.max_iter}")
                task_func = MandelbrotExplorer._gpu_calculation_task if self.use_gpu else MandelbrotExplorer._cpu_calculation_task
                future = self.executor.submit( task_func, current_params['cx'], current_params['cy'],
                                               current_params['zoom'], current_params['width'],
                                               current_params['height'], current_params['max_iter'],
                                               self.current_render_id )
                future.add_done_callback(self._calculation_done_callback)
                self.calculation_future = future
            else:
                self.needs_update = False

    # handle_event: Removed zoom limit
    def handle_event(self, event):
        if event.type == pygame.QUIT: return False
        elif event.type == pygame.VIDEORESIZE:
            # ... (resize handling unchanged) ...
            old_width, old_height = self.width, self.height; self.width, self.height = event.w, event.h
            if self.width <= 0 or self.height <= 0: self.width, self.height = old_width, old_height; return True
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
            self._request_update()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # ... (pan start unchanged) ...
            self.panning = True; self.pan_start_pos = event.pos; self.pan_start_center = (self.center_x, self.center_y)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            # ... (pan end unchanged) ...
            self.panning = False; self.pan_start_pos = None; self.pan_start_center = None
        elif event.type == pygame.MOUSEMOTION and self.panning:
            # ... (pan motion unchanged) ...
            dx = event.pos[0] - self.pan_start_pos[0]; dy = event.pos[1] - self.pan_start_pos[1]
            scale = self.view_width / self.width
            self.center_x = self.pan_start_center[0] - dx * scale; self.center_y = self.pan_start_center[1] + dy * scale
            self._request_update()
        elif event.type == pygame.MOUSEWHEEL:
            mouse_pos = pygame.mouse.get_pos(); mouse_complex_before = self._pixel_to_complex(mouse_pos[0], mouse_pos[1])
            zoom_factor = 1.2 if event.y > 0 else 1 / 1.2; new_zoom = self.zoom * zoom_factor

            # --- Removed Zoom Limit ---
            # if new_zoom > 1e14: print("Zoom limit reached."); return True # REMOVED
            print(f"Zoom: {new_zoom:.3e}") # Log zoom level

            self.zoom = new_zoom; self.view_width = 4.0 / self.zoom
            mouse_complex_after_zoom_no_center_shift = self._pixel_to_complex(mouse_pos[0], mouse_pos[1])
            self.center_x += mouse_complex_before.real - mouse_complex_after_zoom_no_center_shift.real
            self.center_y += mouse_complex_before.imag - mouse_complex_after_zoom_no_center_shift.imag

            if ADAPTIVE_ITER:
                # Adjusted adaptive formula slightly
                new_max_iter = max(50, min(8000, int(INITIAL_MAX_ITER + 60 * math.log10(max(1, self.zoom))))) # Higher factor/cap
                if new_max_iter != self.max_iter:
                    self.max_iter = new_max_iter; print(f"Adjusted max_iter to {self.max_iter}")
            self._request_update()
        return True

    # draw remains the same
    def draw(self):
        self.screen.fill((0, 10, 20)); blit_x = (self.screen.get_width() - self.mandel_surface.get_width()) // 2
        blit_y = (self.screen.get_height() - self.mandel_surface.get_height()) // 2
        self.screen.blit(self.mandel_surface, (blit_x, blit_y))
        info_text = [ f"Backend: {'CUDA Smooth' if self.use_gpu else 'CPU Parallel Smooth'}",
                      f"Center: ({self.center_x:.6g}, {self.center_y:.6g})",
                      f"Zoom: {self.zoom:.3e}", f"Max Iter: {self.max_iter}" ]
        y_offset = 5
        for i, line in enumerate(info_text):
            text_surface = self.font.render(line, True, (255, 255, 255), (0, 0, 0, 180))
            self.screen.blit(text_surface, (5, y_offset + i * 20))
        pygame.display.flip()

    # run remains the same
# run method corrected
    def run(self):
        running = True
        while running: # Outer loop
            events=pygame.event.get() # Get events
            for event in events: # Inner loop (process current events)
                running=self.handle_event(event) # Handle one event, update running flag
                if not running:
                    break # Exit the inner 'for event' loop
            # ----> ADD THIS CHECK <----
            # After processing events, check if we need to exit the outer 'while' loop
            if not running:
                break # Exit the outer 'while running' loop
            # --------------------------

            # If still running, proceed with calculation and drawing
            self._start_calculation()
            self.draw()
            self.clock.tick(60) # Limit FPS

        # Cleanup code runs after the while loop exits
        print("Shutting down...")
        self.executor.shutdown(wait=True)
        pygame.quit()
        print("Exited.")

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Initializing Mandelbrot Explorer...")
    explorer = MandelbrotExplorer(WIDTH, HEIGHT, INITIAL_MAX_ITER, COLORMAP_NAME)
    print(f"Using {'GPU (CUDA Smooth)' if explorer.use_gpu else 'CPU (Parallel Smooth)'} backend.")
    if ADAPTIVE_ITER: print(f"Adaptive iterations enabled, starting at {explorer.max_iter}.")
    else: print(f"Adaptive iterations disabled, fixed at {explorer.max_iter}.")
    print("Note: Zooming beyond approx 1e16 may show precision artifacts (pixelation).") # Added warning
    explorer.run()