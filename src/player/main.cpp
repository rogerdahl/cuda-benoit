// Benoit Track Player. 2011. dahlsys.com
// http://www.dahlsys.com/software/benoit/index.html
// License: GPL.
//
// This project loosely follows the Google style guide:
// https://google-styleguide.googlecode.com/svn/trunk/cppguide.xml

#include "config.h"
#include "cuda_timers.h"
#include "cuda_util.h"
#include "kernels.h"
#include "pch.h"
#include "platform.h"
#include "track.h"


using namespace std;

// Config.

// Size of timer bars.
const u32 BAR_HEIGHT(12);
const u32 BAR_GAP(2);
const u32 NOTCH_HEIGHT(10);

// Globals.

GLuint g_display_tex;
cudaGraphicsResource* g_graphics_resources[1];
Configuration g_cfg;
CUDATimers* g_cuda_timers;
u32 mouse_buttons(0);
u32 g_refresh_rate(0);
u32 g_cuda_device;

// Declarations.

int main(int argc, char** argv);
bool InitGL(int* argc, char** argv);
bool InitCUDA(int argc, char** argv);
StaticTracks* ReadAllTracks();
void CreateAndRegisterTex(
    GLuint& tex, cudaGraphicsResource*& resource, u32 w, u32 h);
void Update();
void Cleanup();
// Rendering callbacks.
void Display();
void Keyboard(unsigned char key, int x, int y);
void Mouse(int button, int state, int x, int y);

// ----------------------------------------------------------------------------
// main.
// ----------------------------------------------------------------------------

int main(int argc, char** argv)
{
  // Read the configuration file.
  if (!read_config(g_cfg)) {
    return 1;
  }

  // Initialize the OpenGL context. The context must exist when CUDA is
  // initialized to get optimal performance with OpenGL/CUDA interop.
  if (!InitGL(&argc, argv)) {
    return 1;
  }

  // Initialize CUDA.
  if (!InitCUDA(argc, argv)) {
    return 1;
  }

  // Read the *.doc tracks from the tracks folder into a static structure.
  StaticTracks* static_tracks(ReadAllTracks());
  if (!static_tracks->count_) {
    cout << "No tracks found" << endl;
    return 1;
  }

  g_refresh_rate = GetDisplayRefreshRate();

  // This app uses a single shared resource between OpenGL and CUDA; the texture
  // that the fractal is rendered into.
  CreateAndRegisterTex(
      g_display_tex, g_graphics_resources[0], g_cfg.screen_w_, g_cfg.screen_h_);

  // Create timers.
  g_cuda_timers = new CUDATimers(kCount_, g_cfg.timers_);

  // Initialize the .cu based graphics resources.
  checkCudaErrors(cudaGraphicsMapResources(1, g_graphics_resources));
  Initialize(1, g_graphics_resources, static_tracks);
  checkCudaErrors(cudaGraphicsUnmapResources(1, g_graphics_resources));

  // Use the ANSI C/C++ atexit() call to specify the address of a function to
  // execute when the program exits.
  atexit(Cleanup);

  // Enter the GLUT event processing loop. This routine should be called at most
  // once in a GLUT program. Once called, this routine will never return. It
  // will call as necessary any callbacks that have been registered.
  glutMainLoop();
}

bool InitGL(int* argc, char** argv)
{
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(g_cfg.screen_w_, g_cfg.screen_h_);
  glutCreateWindow("CUDA Benoit Player");
  if (g_cfg.fullscreen_) {
    glutFullScreen();
    // Hide the mouse pointer in full screen mode.
    glutSetCursor(GLUT_CURSOR_NONE);
  }

  // Set vsync and vsync interval.
  // typedef BOOL (WINAPI * PFNWGLSWAPINTERVALEXTPROC) (int interval);
  // PFNWGLSWAPINTERVALEXTPROC	wglSwapInterval;
  // wglSwapInterval =
  // (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
  // wglSwapInterval(g_cfg.vsync_ ? g_cfg.vsync_interval_ : 0);

  // Register callbacks.
  glutDisplayFunc(Display);
  glutKeyboardFunc(Keyboard);
  glutMouseFunc(Mouse);

  // Initialize necessary OpenGL extensions.
  glewInit();
  if (!glewIsSupported("GL_VERSION_2_0 ")) {
    fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing");
    fflush(stderr);
    return false;
  }

  // Default color.
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

  // 2D projection.
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, g_cfg.screen_w_, g_cfg.screen_h_, 0, 0, 1);

  // No depth in 2D.
  glDisable(GL_DEPTH_TEST);

  // GL_MODELVIEW matrix is a combination of Model and View matrices (Mview *
  // Mmodel). Model transform is to convert from object space to world space.
  // View transform is to convert from world space to eye space.
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // Displacement trick for exact pixelization. Causes blurry image, probably
  // because pixelization is already being taken into account in the kernels.
  // glTranslatef(0.375, 0.375, 0);

  // Exit if anything above caused an error.
  GLenum gl_error(glGetError());
  if (gl_error != GL_NO_ERROR) {
    return false;
  }

  return true;
}

bool InitCUDA(int argc, char** argv)
{
  // Use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s.
  if (g_cfg.device_ == -1) {
    // g_cuda_device = cutGetMaxGflopsDeviceId();
    g_cuda_device = 0;
    cudaSetDevice(g_cuda_device);
    cudaGLSetGLDevice(g_cuda_device);
  }
  else {
    g_cuda_device = g_cfg.device_;
    GLDeviceInit(g_cuda_device);
  }
  // Shared memory is not used in any of the kernels in this app (yet), so set
  // the global cache preference to L1.
  cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
  return true;
}

// Create a texture and register it for sharing with CUDA.
void CreateAndRegisterTex(
    GLuint& tex, cudaGraphicsResource*& resource, u32 w, u32 h)
{
  u32 i(0);
  uchar4* buf((uchar4*)malloc(w * h * sizeof(uchar4)));
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  //  glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // needed?
  glTexImage2D(GL_TEXTURE_2D, 0, 4, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, buf);
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_REPEAT);
  glBindTexture(GL_TEXTURE_2D, 0);
  free(buf);
  glBindTexture(GL_TEXTURE_2D, tex);
  // Register this image for sharing with CUDA. cudaGraphicsGLRegisterImage()
  // supports all texture formats with 1, 2, or 4 components and an internal
  // type of float (e.g. GL_RGBA_FLOAT32) and unnormalized integer (e.g.
  // GL_RGBA8UI). It does not currently support normalized integer formats (e.g.
  // GL_RGBA8). Please note that since GL_RGBA8UI is an OpenGL 3.0 texture
  // format, it can only be written by shaders, not the fixed function pipeline.
  checkCudaErrors(
      cudaGraphicsGLRegisterImage(
          &resource, tex, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
}

// Read all tracks and store them in a static structure.
StaticTracks* ReadAllTracks()
{
  // const u32 path_buf_size(2048);
  // char exe[path_buf_size];
  // GetModuleFileName(NULL, exe, path_buf_size);
  // string exe_str(exe);
  // boost::filesystem::path track_dir_path(exe_str.substr(0,
  // exe_str.rfind("\\")) + "\\tracks");
  return ReadTracks(boost::filesystem::path("./tracks"));
}

void Update()
{
  // Make the OpenGL texture available to CUDA.
  //
  // Some driver bug causes this call to be extremely slow on a 2 GPU system
  // even when both CUDA and OpenGL runs on the same GPU.
  checkCudaErrors(cudaGraphicsMapResources(1, g_graphics_resources));
  // Calculate a new frame and update the texture with it.
  bool mouse_button_left((mouse_buttons & 1) != 0);
  bool mouse_button_right((mouse_buttons & 4) != 0);
  FractalCalc(mouse_button_left, mouse_button_right);
  // Unmap the texture so that it can be used for rendering in OpenGL.
  checkCudaErrors(cudaGraphicsUnmapResources(1, g_graphics_resources));
}

// Draw fractal.
void DrawFractal()
{
  // Draw a single textured quad to display the fractal.
  glEnable(GL_TEXTURE_2D);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glBindTexture(GL_TEXTURE_2D, g_display_tex);
  // Draw counter-clockwise.
  glBegin(GL_QUADS);
  // Bottom Left.
  glTexCoord2f(0.0f, 1.0f);
  glVertex2i(0, g_cfg.screen_h_);
  // Bottom Right.
  glTexCoord2f(1.0f, 1.0f);
  glVertex2i(g_cfg.screen_w_, g_cfg.screen_h_);
  // Top Right.
  glTexCoord2f(1.0f, 0.0f);
  glVertex2i(g_cfg.screen_w_, 0);
  // Top Left.
  glTexCoord2f(0.0f, 0.0f);
  glVertex2i(0, 0);
  glEnd();
  // Unbind.
  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_TEXTURE_2D);
}

// Draw a text string.
void DrawString(int pos_x, int pos_y, char* str)
{
  glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
  for (u32 i(0);; ++i) {
    char c(str[i]);
    if (!c) {
      return;
    }
    glRasterPos2i(pos_x, pos_y);
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c);
    pos_x += glutBitmapWidth(GLUT_BITMAP_HELVETICA_12, c);
  }
}

// Draw the notch that designates the time available for rendering a frame. It
// is centered horizontally to enable easy representation of 0 to up to 2x the
// available time.
void DrawTimerNotch()
{
  float center(static_cast<float>(g_cfg.screen_w_) / 2.0);
  glColor3f(1.0f, 0.0f, 0.0f);
  glBegin(GL_TRIANGLES);
  glVertex2i(center - 5, 0);
  glVertex2i(center + 5, 0);
  glVertex2i(center, NOTCH_HEIGHT);
  glEnd();
}

// Draw a timer mark. These are the small black marks that designate the
// average, min and max values for each timer bar.
void DrawTimerMark(u32 i, double frame_time, double time)
{
  double bar_w(time / frame_time * static_cast<double>(g_cfg.screen_w_) / 2.0);
  glColor3f(0, 0, 0);
  glBegin(GL_QUADS);
  u32 p1(NOTCH_HEIGHT + i * (BAR_HEIGHT + BAR_GAP));
  u32 p2(p1 + BAR_HEIGHT);
  glVertex2i(bar_w - 2, p1);
  glVertex2i(bar_w + 2, p1);
  glVertex2i(bar_w + 2, p2);
  glVertex2i(bar_w - 2, p2);
  glEnd();
}

// Draw a timer bar.
void DrawTimerBar(
    u32 i, const char* label, double frame_time, const CUDATimes& times)
{
  double bar_w(
      times.current_ / frame_time * static_cast<double>(g_cfg.screen_w_) / 2.0);
  // Draw bar.
  glColor3f(0.5f, 0.5f, 0.5f);
  glBegin(GL_QUADS);
  u32 p1(NOTCH_HEIGHT + i * (BAR_HEIGHT + BAR_GAP));
  u32 p2(p1 + BAR_HEIGHT);
  glVertex2i(0, p1);
  glVertex2i(bar_w, p1);
  glVertex2i(bar_w, p2);
  glVertex2i(0, p2);
  glEnd();
  // Draw average, min and max marks.
  DrawTimerMark(i, frame_time, times.average_);
  DrawTimerMark(i, frame_time, times.min_);
  DrawTimerMark(i, frame_time, times.max_);
  // Draw label.
  char format_buf[128];
  sprintf(format_buf, "%s %.1fms", label, times.average_ * 1000.0);
  DrawString(10, p2 - (BAR_GAP / 2), format_buf);
}

// Clean up resources before exit.
void Cleanup()
{
  delete g_cuda_timers;
  cudaThreadExit();
  // The app exits here.
}

// ----------------------------------------------------------------------------
// GLUT callbacks.
// ----------------------------------------------------------------------------

// Display event handler.
void Display()
{
  {
    CUDATimerRun run_total_timer(*g_cuda_timers, kTotal);

    // Clear color and z buffer.
    glClear(GL_COLOR_BUFFER_BIT);

    // Generate new frame.
    Update();

    // Render.
    {
      CUDATimerRun run_render_timer(*g_cuda_timers, kRender);
      DrawFractal();
    }
  }

  // Draw timers.
  if (g_cfg.timers_) {
    double frame_time(1.0 / static_cast<double>(g_refresh_rate));
    if (g_cfg.vsync_) {
      frame_time *= g_cfg.vsync_interval_;
    }
    u32 pos(0);
    DrawTimerNotch();
    DrawTimerBar(pos++, "total", frame_time, g_cuda_timers->GetTimes(kTotal));
    DrawTimerBar(
        pos++, "mandelbrot", frame_time, g_cuda_timers->GetTimes(kMandelbrot));
    DrawTimerBar(
        pos++, "fractal reduce", frame_time,
        g_cuda_timers->GetTimes(kFractalReduce));
    DrawTimerBar(
        pos++, "palettes", frame_time, g_cuda_timers->GetTimes(kPalettes));
    DrawTimerBar(
        pos++, "log transform", frame_time,
        g_cuda_timers->GetTimes(kTransform));
    DrawTimerBar(
        pos++, "reduce and colorize", frame_time,
        g_cuda_timers->GetTimes(kTransformReduceAndColorize));
    DrawTimerBar(pos++, "render", frame_time, g_cuda_timers->GetTimes(kRender));
  }

  glutSwapBuffers();
  glutPostRedisplay();
}

// Keyboard events handler.
void Keyboard(unsigned char key, int /*x*/, int /*y*/)
{
  switch (key) {
  // Esc key
  case (27):
    // Issuing exit() causes Cleanup() to get called (it was registered with
    // atexit()), after which the app exits.
    exit(0);
    break;
  }
}

// Mouse events handler.
void Mouse(int button, int state, int x, int y)
{
  if (state == GLUT_DOWN) {
    mouse_buttons |= 1 << button;
  }
  else if (state == GLUT_UP) {
    mouse_buttons &= ~(1 << button);
  }
}
