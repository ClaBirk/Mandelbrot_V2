# --- Keep imports the same ---
import pygame
import numpy as np
import matplotlib
import matplotlib.colors as colors
from concurrent.futures import ProcessPoolExecutor
from numba import jit, NumbaWarning
import time
import sys
import traceback
import warnings
import math

# --- Configuration ---
WIDTH, HEIGHT = 800, 600
INITIAL_MAX_ITER = 150
ADAPTIVE_ITER = True
COLORMAP_NAME = 'plasma' # Store colormap name

# warnings.simplefilter('ignore', category=NumbaWarning)

# --- Numba Calculation Function (Unchanged) ---
@jit(nopython=True, fastmath=True, cache=True)
def compute_mandelbrot_iterations(c_real, c_imag, max_iter):
    # ... (function remains the same as the previous version) ...
    iterations = np.zeros_like(c_real, dtype=np.uint32)
    z_real = np.zeros_like(c_real); z_imag = np.zeros_like(c_imag)
    active = np.ones(c_real.shape, dtype=np.bool_)
    max_iter_uint32 = np.uint32(max_iter)
    # --- Interior Checking ---
    cr_minus_1_4 = c_real - 0.25
    q = cr_minus_1_4 * cr_minus_1_4 + c_imag * c_imag
    quarter_y_sq = 0.25 * c_imag * c_imag
    inside_cardioid = np.less(q * (q + cr_minus_1_4), quarter_y_sq)
    cr_plus_1 = c_real + 1.0
    inside_bulb2 = np.less(cr_plus_1 * cr_plus_1 + c_imag * c_imag, 0.0625)
    is_inside = np.logical_or(inside_cardioid, inside_bulb2)
    iterations = np.where(is_inside, max_iter_uint32, iterations)
    active = np.where(is_inside, False, active)
    # --- End Interior Checking ---
    for n in range(max_iter):
        active_mask_this_iter = np.copy(active)
        if not np.any(active_mask_this_iter): break
        z_real_sq = z_real * z_real; z_imag_sq = z_imag * z_imag
        z_real_imag = z_real * z_imag
        temp_zr_new = z_real_sq - z_imag_sq + c_real
        temp_zi_new = 2.0 * z_real_imag + c_imag
        z_real = np.where(active_mask_this_iter, temp_zr_new, z_real)
        z_imag = np.where(active_mask_this_iter, temp_zi_new, z_imag)
        magnitude_sq = z_real * z_real + z_imag * z_imag
        escaped_now = magnitude_sq > 4.0
        just_escaped = active_mask_this_iter & escaped_now
        if np.any(just_escaped):
            iterations = np.where(just_escaped, np.uint32(n + 1), iterations)
            active = np.logical_and(active, ~just_escaped)
    iterations = np.where(active, max_iter_uint32, iterations)
    return iterations

# --- Color Mapping Function Generator (Unchanged) ---
def get_colormap_func(cmap_name):
    """Returns a function that applies colormap for a given max_iter."""
    # This function itself is defined at the top level, so it's pickleable.
    # The function *it returns* was the problem.
    try:
        cmap = matplotlib.colormaps[cmap_name]
    except AttributeError:
        import matplotlib.cm as cm
        print("Warning: Using deprecated cm.get_cmap(). Update Matplotlib if possible.")
        cmap = cm.get_cmap(cmap_name)

    def apply_colormap(iterations, current_max_iter):
        # This inner function is NOT passed between processes anymore.
        norm = colors.Normalize(vmin=0, vmax=current_max_iter)
        max_iter_uint32_cmp = np.uint32(current_max_iter)
        inside_mask = iterations == max_iter_uint32_cmp
        rgba_colors = cmap(norm(iterations))
        rgb_colors = (rgba_colors[:, :, :3] * 255).astype(np.uint8)
        rgb_colors[inside_mask] = [0, 0, 0]
        return np.ascontiguousarray(rgb_colors)

    return apply_colormap


# --- Main Application Class (Pickling Fix) ---
class MandelbrotExplorer:
    def __init__(self, width, height, initial_max_iter, colormap_name):
        pygame.init()
        pygame.display.set_caption("Interactive Mandelbrot Explorer (Multiprocessing Fixed)")
        self.width = width; self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self.font = pygame.font.SysFont(None, 24); self.clock = pygame.time.Clock()
        self.center_x = -0.7; self.center_y = 0.0; self.zoom = 1.0
        self.view_width = 4.0 / self.zoom
        self.max_iter = initial_max_iter
        self.colormap_name = colormap_name # Store name instead of function

        self.mandel_surface = pygame.Surface((self.width, self.height)); self.mandel_surface.fill((0,0,0))
        # self.apply_colormap = get_colormap_func(colormap_name) # No longer needed here

        self.panning = False; self.pan_start_pos = None; self.pan_start_center = None
        self.executor = ProcessPoolExecutor(max_workers=None)
        self.calculation_future = None; self.needs_update = True
        self.last_calc_params = None; self.current_render_id = 0; self.next_render_id = 0
        self.CALC_DONE_EVENT = pygame.USEREVENT + 1 # Not used with callbacks, but harmless

    # _get_complex_coords, _pixel_to_complex remain the same
    def _get_complex_coords(self):
        scale = self.view_width / self.width
        real_min=self.center_x-(self.width/2)*scale; real_max=self.center_x+(self.width/2)*scale
        imag_min=self.center_y-(self.height/2)*scale; imag_max=self.center_y+(self.height/2)*scale
        c_real_1d = np.linspace(real_min, real_max, self.width, dtype=np.float64)
        c_imag_1d = np.linspace(imag_max, imag_min, self.height, dtype=np.float64)
        c_real, c_imag = np.meshgrid(c_real_1d, c_imag_1d)
        return np.ascontiguousarray(c_real), np.ascontiguousarray(c_imag)

    def _pixel_to_complex(self, px, py):
        scale = self.view_width / self.width
        real = self.center_x + (px - self.width / 2) * scale
        imag = self.center_y + (self.height / 2 - py) * scale
        return complex(real, imag)

    # Static method for calculation task
    # Now takes cmap_name instead of colormap_func
    @staticmethod
    def _calculation_task(cx, cy, zoom, width, height, max_iter, render_id, cmap_name):
        try:
            start_time = time.time()
            print(f"Starting calculation ID {render_id} ({width}x{height}, iter={max_iter})...")

            # Calculate coordinates
            view_width = 4.0 / zoom; scale = view_width / width
            real_min=cx-(width/2)*scale; real_max=cx+(width/2)*scale
            imag_min=cy-(height/2)*scale; imag_max=cy+(height/2)*scale
            c_real_1d=np.linspace(real_min,real_max,width,dtype=np.float64)
            c_imag_1d=np.linspace(imag_max,imag_min,height,dtype=np.float64)
            c_real,c_imag=np.meshgrid(c_real_1d,c_imag_1d)
            c_real,c_imag=np.ascontiguousarray(c_real),np.ascontiguousarray(c_imag)

            # Compute iterations
            iterations = compute_mandelbrot_iterations(c_real, c_imag, max_iter)

            # --- Get colormap function LOCALLY in the worker ---
            apply_colormap_local = get_colormap_func(cmap_name)
            rgb_array = apply_colormap_local(iterations, max_iter)
            # ----------------------------------------------------

            end_time = time.time(); calc_time = end_time - start_time
            print(f"Calculation ID {render_id} finished in {calc_time:.3f}s")

            return { 'rgb_array': rgb_array, 'render_id': render_id, 'calc_time': calc_time,
                     'params': {'cx': cx, 'cy': cy, 'zoom': zoom, 'width': width, 'height': height, 'max_iter': max_iter} }
        except Exception as e:
            print(f"Error in calculation process (ID {render_id}): {e}")
            traceback.print_exc()
            return {'error': True, 'render_id': render_id}

    # Callback function remains the same
    def _calculation_done_callback(self, future):
        try:
            result = future.result()
            if result.get('error', False):
                 print(f"Calculation task ID {result.get('render_id', 'unknown')} failed.")
                 return
            if result['render_id'] == self.current_render_id:
                params = result['params']
                if params['width'] == self.width and params['height'] == self.height:
                    print(f"Received data for ID {result['render_id']}")
                    rgb_array = result['rgb_array']
                    surface = pygame.surfarray.make_surface(np.transpose(rgb_array, (1, 0, 2)))
                    self.mandel_surface = surface
                    self.last_calc_params = params
                else:
                    print(f"Warning: Received data size != window size. Requesting update.")
                    self._request_update()
            else:
                print(f"Ignoring stale calculation result (ID {result['render_id']}, expected {self.current_render_id})")
        except Exception as e:
            print(f"Error processing calculation result: {e}")
            traceback.print_exc()

    # _request_update remains the same
    def _request_update(self):
        self.needs_update = True; self.next_render_id += 1

    # _start_calculation needs to pass cmap_name
    def _start_calculation(self):
        if self.needs_update:
            current_params = { 'cx': self.center_x, 'cy': self.center_y, 'zoom': self.zoom,
                               'width': self.width, 'height': self.height, 'max_iter': self.max_iter }
            if current_params != self.last_calc_params:
                self.needs_update = False
                self.current_render_id = self.next_render_id
                print(f"Submitting calculation ID {self.current_render_id} with max_iter={self.max_iter}")

                # Submit task to ProcessPoolExecutor
                # Pass self.colormap_name instead of the function
                future = self.executor.submit(
                    MandelbrotExplorer._calculation_task, # Static method
                    current_params['cx'], current_params['cy'], current_params['zoom'],
                    current_params['width'], current_params['height'], current_params['max_iter'],
                    self.current_render_id,
                    self.colormap_name # Pass the name (string)
                )
                future.add_done_callback(self._calculation_done_callback)
                self.calculation_future = future
            else:
                self.needs_update = False

    # handle_event remains the same
    def handle_event(self, event):
        if event.type == pygame.QUIT: return False
        elif event.type == pygame.VIDEORESIZE:
            old_width, old_height = self.width, self.height; self.width, self.height = event.w, event.h
            if self.width <= 0 or self.height <= 0:
                self.width, self.height = old_width, old_height; return True
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
            self._request_update()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.panning = True; self.pan_start_pos = event.pos; self.pan_start_center = (self.center_x, self.center_y)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.panning = False; self.pan_start_pos = None; self.pan_start_center = None
        elif event.type == pygame.MOUSEMOTION and self.panning:
            dx = event.pos[0] - self.pan_start_pos[0]; dy = event.pos[1] - self.pan_start_pos[1]
            scale = self.view_width / self.width
            self.center_x = self.pan_start_center[0] - dx * scale; self.center_y = self.pan_start_center[1] + dy * scale
            self._request_update()
        elif event.type == pygame.MOUSEWHEEL:
            mouse_pos = pygame.mouse.get_pos(); mouse_complex_before = self._pixel_to_complex(mouse_pos[0], mouse_pos[1])
            zoom_factor = 1.2 if event.y > 0 else 1 / 1.2; new_zoom = self.zoom * zoom_factor
            if new_zoom > 1e14: print("Zoom limit reached."); return True
            self.zoom = new_zoom; self.view_width = 4.0 / self.zoom
            mouse_complex_after_zoom_no_center_shift = self._pixel_to_complex(mouse_pos[0], mouse_pos[1])
            self.center_x += mouse_complex_before.real - mouse_complex_after_zoom_no_center_shift.real
            self.center_y += mouse_complex_before.imag - mouse_complex_after_zoom_no_center_shift.imag
            if ADAPTIVE_ITER:
                new_max_iter = max(INITIAL_MAX_ITER, min(2000, int(INITIAL_MAX_ITER + 30 * math.log10(max(1, self.zoom)))))
                if new_max_iter != self.max_iter:
                    self.max_iter = new_max_iter; print(f"Adjusted max_iter to {self.max_iter}")
            self._request_update()
        return True

    # draw remains the same
    def draw(self):
        self.screen.fill((0, 10, 20)); blit_x = (self.screen.get_width() - self.mandel_surface.get_width()) // 2
        blit_y = (self.screen.get_height() - self.mandel_surface.get_height()) // 2
        self.screen.blit(self.mandel_surface, (blit_x, blit_y))
        info_text = [ f"Center: ({self.center_x:.6g}, {self.center_y:.6g})",
                      f"Zoom: {self.zoom:.3e}", f"Max Iter: {self.max_iter}" ]
        y_offset = 5
        for i, line in enumerate(info_text):
            text_surface = self.font.render(line, True, (255, 255, 255), (0, 0, 0, 180))
            self.screen.blit(text_surface, (5, y_offset + i * 20))
        pygame.display.flip()

    # run remains the same
    def run(self):
        running = True
        while running:
            events = pygame.event.get();
            for event in events:
                running = self.handle_event(event)
                if not running: break
            if not running: break
            self._start_calculation(); self.draw(); self.clock.tick(60)
        print("Shutting down...")
        self.executor.shutdown(wait=True); pygame.quit()
        print("Exited.")

# --- Main Execution (Needs if __name__ == "__main__":) ---
if __name__ == "__main__":
    # Set Pygame logging level to reduce INFO messages if desired
    # import os
    # os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1" # Doesn't always work reliably
    # pygame.init() # Init pygame *after* setting env var maybe? No, init is in Explorer class.

    print(f"Note: Using ProcessPoolExecutor for parallel calculation.")
    if ADAPTIVE_ITER:
        print(f"Adaptive iterations enabled, starting at {INITIAL_MAX_ITER}.")
    else:
        print(f"Adaptive iterations disabled, fixed at {INITIAL_MAX_ITER}.")

    # Pass the colormap NAME to the explorer
    explorer = MandelbrotExplorer(WIDTH, HEIGHT, INITIAL_MAX_ITER, COLORMAP_NAME)
    explorer.run()