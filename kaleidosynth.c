#include <stdio.h>
#include <math.h>
#include <portaudio.h>
#include <complex.h>
#include <signal.h>
#include "kiss_fftr.h"

typedef int retcode;
#define retfail(CONDITION) { retcode __retval = CONDITION ; if((__retval < 0)) \
  { fprintf(stderr, "Failed: %s\n In %s on line %d - %d\n", #CONDITION, __FILE__, __LINE__, __retval ); return __retval; } }

#define TABLE_SIZE 2048
typedef struct {
  kiss_fft_scalar buffer_data[TABLE_SIZE]; // pre-buffer size
  kiss_fft_scalar freq_domain[TABLE_SIZE]; // pre-buffer size
  int left_phase;
  int note_pos;
  kiss_fftr_cfg cfg;
} LRAudioBuf;
  
/// GLOBALS ///  
LRAudioBuf audio_buf = { 0 };
PaStream *stream = NULL;

// This can be called at interrupt level, so nothing fancy, no malloc/free
static int audio_buffer_sync_callback(
    const void *inputBuffer, // unused
    void *outputBuffer,
    unsigned long framesPerBuffer, // This tends to be 64
    const PaStreamCallbackTimeInfo* timeInfo, // unused
    PaStreamCallbackFlags stats, // unused
    void *context ) {

  LRAudioBuf *audio_buf = (LRAudioBuf*)context;
  float *out = (float*)outputBuffer;
  unsigned long i;

  for( i=0; i<framesPerBuffer; i++ ) {
    *out++ = audio_buf->buffer_data[audio_buf->left_phase];  // left channel
    audio_buf->left_phase += 1;
    if( audio_buf->left_phase >= TABLE_SIZE ) {
      audio_buf->left_phase -= TABLE_SIZE;
      audio_buf->freq_domain[audio_buf->note_pos] = 0;
      audio_buf->freq_domain[TABLE_SIZE - audio_buf->note_pos -1] = 0;
      audio_buf->note_pos = (audio_buf->note_pos + 8) % (TABLE_SIZE);
      audio_buf->freq_domain[audio_buf->note_pos] = 32;
      audio_buf->freq_domain[TABLE_SIZE - audio_buf->note_pos -1] = -32;
      kiss_fftri(audio_buf->cfg, (kiss_fft_cpx *)audio_buf->freq_domain, audio_buf->buffer_data);
    }
  }
  return paContinue;
}

static void cleanup(void *context) {
  LRAudioBuf *audio_buf = (LRAudioBuf*)context;
  free(audio_buf->cfg);
}
  
int near(float a, float b, float epsilon) {
  return (a + epsilon > b && a - epsilon < b);
}

static int init_portaudio() {
  PaStreamParameters outputParameters = { 0 };
  memset(&outputParameters, 0, sizeof(outputParameters));
  
  // init the buzz with a real ifft
  memset(&audio_buf, 0, sizeof(audio_buf));
  audio_buf.cfg = kiss_fftr_alloc(TABLE_SIZE, 1, NULL,NULL);

  retfail(Pa_Initialize());

  outputParameters.device = Pa_GetDefaultOutputDevice(); /* default output device */
  retfail(outputParameters.device == paNoDevice) // no default output device
  
  outputParameters.channelCount = 1; // If this is stereo, then the autputBuffer is (L, R) tuples
  outputParameters.sampleFormat = paFloat32; /* 32 bit floating point output */
  outputParameters.suggestedLatency = Pa_GetDeviceInfo( outputParameters.device )->defaultLowOutputLatency;
  outputParameters.hostApiSpecificStreamInfo = NULL;

  retfail(Pa_OpenStream(
      &stream,
      NULL, // no input
      &outputParameters,
      44100, // sample rate
      64, // sample frames per buffer
      paClipOff,
      audio_buffer_sync_callback,
      &audio_buf )); // this is context in the callback
  
  retfail(Pa_SetStreamFinishedCallback(stream, &cleanup));

  return 0;
}

static int shutdown() {
    retfail(Pa_StopStream( stream ));
    retfail(Pa_CloseStream( stream ));
    return 0;
}

void sighandler(int signo) {
  if (signo == SIGKILL) {
    printf("Shutting down...");
    shutdown();
  }
}

#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>

#define FPS 60
#define HEIGHT 640
#define WIDTH 480
// RGB = 4
#define COLOURS 3
void display() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_TEXTURE_2D);

  unsigned char framebuffer[HEIGHT][WIDTH][COLOURS] = { 0 };

  for(int i=0; i < HEIGHT; ++i) {
    for(int j=0; j < WIDTH; ++j) {
      framebuffer[i][j][1] = (unsigned char) j;
      framebuffer[i][j][2] = (unsigned char) i;
      //framebuffer[i][j][3] = (unsigned char) j + i;
    }
  }
  int tex_id = 0;

  glGenTextures(1, &tex_id);
  glBindTexture(GL_TEXTURE_2D, tex_id);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, HEIGHT, WIDTH, 0, GL_RGB,
               GL_UNSIGNED_BYTE, framebuffer);
  //glPolygonMode(GL_FRONT_AND_BACK, self->polygon_mode);

  glBegin(GL_QUADS);
    glTexCoord2f (0.5f,0.5f);
    glVertex3f(-1,-1,0);
    
    glTexCoord2f (0.0f,0.5f);
    glVertex3f(1,-1,0);
    
    glTexCoord2f (0.0f,0.0f);
    glVertex3f(1,1,0);
    
    glTexCoord2f (0.5f,0.0f);
    glVertex3f(-1,1,0);
  glEnd();

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  gluOrtho2D(0, 1, 0, 1);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glRasterPos3f(0.0, 0.0, 0);

  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);

  glutSwapBuffers(); /* calls glFlush() */
}

static volatile int frame_count = 0;
static const int SECONDS = 3;
void timer(int value) {
  glutPostRedisplay();
  glutTimerFunc(1000 / FPS, &timer, value);
  frame_count ++;
  if(frame_count > 60 * SECONDS) {
    // cleanup sound //
    shutdown();
    printf("All good.\n");
    exit(0);
  }
}

int main(int argc, char **argv) {
  printf("Hello sound\n");
  retfail(init_portaudio());
  retfail(signal(SIGINT, sighandler) == SIG_ERR);
  retfail(Pa_StartStream(stream));
  
  printf("Hello video\n");
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
  glutInitWindowSize(800, 600);

  glutCreateWindow("Kaleidosynth");
  //glutFullScreen();

  glutDisplayFunc(&display);
  glutTimerFunc(0, &timer, 0);
  glutMainLoop(); // never returns
  return 0;
}
