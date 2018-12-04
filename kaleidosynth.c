#include <stdio.h>
#include <math.h>
#include <portaudio.h>
#include <complex.h>
#include <signal.h>
#include "kiss_fftr.h"
#include "err.h"
#include "nn.h"
#include "gl.h"
#include <time.h>

/// AUDIO GLOBALS ///  
#define AUDIO_BAND WIDTH
typedef struct {
  kiss_fft_scalar buffer_data[AUDIO_BAND]; // pre-buffer size
  kiss_fft_scalar copy_buf[AUDIO_BAND]; // pre-buffer size
  kiss_fft_scalar freq_data[AUDIO_BAND]; // pre-buffer size
  kiss_fft_scalar melody[AUDIO_BAND]; // pre-buffer size
  int left_phase;
  int note_pos;
  int maxpos;
  kiss_fftr_cfg cfg;
} LRAudioBuf;
LRAudioBuf audio_buf = { 0 };
PaStream *stream = NULL;
static const float volumeMultiplier = 0.01f;
static const float SAMPLE_RATE = 44100;

static volatile clock_t lasttime = 0;

/// NEURAL NETWORK GLOBALS ///
static const int nn_input_size = INPUT_DIM; // x, y, frame
static const int nn_batch_size = WIDTH * HEIGHT;
static const int hidden_neurons = 20, output_neurons = COLOURS;
static const int epochs = 10;
static const int num_layers = 4; // THIS MUST MATCH BELOW
static const int last_layer = num_layers -1;
static const float initialization_sigma = 4 / num_layers;
struct neural_layer cppn[] = {
  {
    .weights = { 0 }, .w_delt = { 0 }, .biases = { 0 }, .b_delt = { 0 },
    .activations = { .x = nn_batch_size, .y = nn_input_size, .e = NULL },
    .zvals = { .x = nn_batch_size, .y = nn_input_size, .e = NULL },
    .activate = NULL, .backprop = NULL,
  }, {
    .weights = { .x = nn_input_size, .y = hidden_neurons, .e = NULL },
    .w_delt = { .x = nn_input_size, .y = hidden_neurons, .e = NULL },
    .biases = { .x = nn_batch_size, .y = hidden_neurons, .e = NULL },
    .b_delt = { .x = nn_batch_size, .y = hidden_neurons, .e = NULL },
    .activations = { .x = nn_batch_size, .y = hidden_neurons, .e = NULL },
    .zvals = { .x = nn_batch_size, .y = hidden_neurons, .e = NULL },
    .activate = &gaussian_activate,
    .backprop = &gaussian_prime,
  }, {
    .weights = { .x = hidden_neurons, .y = hidden_neurons, .e = NULL },
    .w_delt = { .x = hidden_neurons, .y = hidden_neurons, .e = NULL },
    .biases = { .x = nn_batch_size, .y = hidden_neurons, .e = NULL },
    .b_delt = { .x = nn_batch_size, .y = hidden_neurons, .e = NULL },
    .activations = { .x = nn_batch_size, .y = hidden_neurons, .e = NULL },
    .zvals = { .x = nn_batch_size, .y = hidden_neurons, .e = NULL },
    .activate = &gaussian_activate,
    .backprop = &gaussian_prime,
  }, {
    .weights = { .x = hidden_neurons, .y = output_neurons, .e = NULL },
    .w_delt = { .x = hidden_neurons, .y = output_neurons, .e = NULL },
    .biases = { .x = nn_batch_size, .y = output_neurons, .e = NULL },
    .b_delt = { .x = nn_batch_size, .y = output_neurons, .e = NULL },
    .activations = { .x = nn_batch_size, .y = output_neurons, .e = NULL },
    .zvals = { .x = nn_batch_size, .y = output_neurons, .e = NULL },
    .activate = &gaussian_activate,
    .backprop = &gaussian_prime,
  },
};

static int seed_network() {
  for (int i = 1; i < num_layers; ++i) {
      randomize(cppn[i].weights.e, 
        cppn[i].weights.x * cppn[i].weights.y, 
        initialization_sigma);
      randomize(cppn[i].biases.e, 
        cppn[i].biases.x * cppn[i].biases.y, 
        initialization_sigma);
    }
}

static int init_neural_network() {
  for (int i = 0; i < num_layers; ++i) {
    if(i == 0) {
      cppn[i].activations.e = malloc(nn_batch_size * nn_input_size * 
        sizeof(float));
      cppn[i].zvals.e = malloc(nn_batch_size * nn_input_size * 
        sizeof(float));
    } else {
      cppn[i].weights.e = malloc(cppn[i].weights.x * cppn[i].weights.y * 
        sizeof(float));
      cppn[i].w_delt.e = calloc(cppn[i].w_delt.x * cppn[i].w_delt.y, 
        sizeof(float));

      cppn[i].biases.e = malloc(cppn[i].biases.x * cppn[i].biases.y * 
        sizeof(float));
      cppn[i].b_delt.e = calloc(cppn[i].b_delt.x * cppn[i].b_delt.y, 
        sizeof(float));
      cppn[i].activations.e = calloc(
        cppn[i].activations.x * cppn[i].activations.y, 
        sizeof(float));
      cppn[i].zvals.e = calloc(cppn[i].zvals.x * cppn[i].zvals.y, 
        sizeof(float));

    }
  }
  seed_network();
  return SUCCESS;
}

/// AUDIO CODE ///
// This can be called at interrupt level, so nothing fancy, no malloc/free
static int audio_buffer_sync_callback(
    const void *inputBuffer, // unused
    void *outputBuffer,
    unsigned long framesPerBuffer, // This is defined in setup 64
    const PaStreamCallbackTimeInfo* timeInfo, // unused
    PaStreamCallbackFlags stats, // unused
    void *context ) {

  LRAudioBuf *audio_buf = (LRAudioBuf*)context;
  float *out = (float*)outputBuffer;
  unsigned long i;

  for( i=0; i<framesPerBuffer/2; i++ ) {
    *out++ = audio_buf->buffer_data[i] 
      * volumeMultiplier;  // left channel
    *out++ = (audio_buf->buffer_data[i] +  audio_buf->buffer_data[(i+1)%AUDIO_BAND])/2
      * volumeMultiplier;  // left channel

    audio_buf->left_phase += 1;
    // refill the audio buffer with the ifft of the visualization scanning
    // down the screen
    if( audio_buf->left_phase >= AUDIO_BAND ) {
      float (*nn_l)[WIDTH][COLOURS] = (void *) 
        cppn[last_layer].activations.e;
      audio_buf->left_phase -= AUDIO_BAND;
      audio_buf->note_pos += 1;
      if (audio_buf->note_pos > HEIGHT) { // SCREENWRAP TIME //
        audio_buf->note_pos = 0;
        if(SHIFT_COLOURS == 0) {
          SHIFT_COLOURS = 1;
        } else {
          SHIFT_COLOURS = 0;
        }
      }
      int p = audio_buf->note_pos;
      
      for (int j=0; j < AUDIO_BAND; ++j) {
        audio_buf->freq_data[j] = nn_l[p][j][0];
        if(COLOURS == 3) {
          audio_buf->freq_data[j] += nn_l[p][j][1] + nn_l[p][j][2];
        }
      }

      kiss_fftri(audio_buf->cfg, 
        (kiss_fft_cpx *) &audio_buf->freq_data,
        audio_buf->buffer_data);
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
  int chosen_device = 0;
  
  // init the buzz with a real ifft
  memset(&audio_buf, 0, sizeof(audio_buf));
  audio_buf.cfg = kiss_fftr_alloc(AUDIO_BAND, 1, NULL,NULL);

  retfail(Pa_Initialize());

  /* default output device */
  outputParameters.device = Pa_GetDefaultOutputDevice(); 
  retfail(outputParameters.device == paNoDevice) 
  int num_devs = Pa_GetDeviceCount();
  PaDeviceInfo *di = NULL;
  for (int i=0; i < num_devs ; ++ i) {
    di = Pa_GetDeviceInfo(i);
    printf("Available Adev: %s\n", di->name);
    if(strcmp(di->name, "default") == 0) {
      printf("Choosing this device\n");
      chosen_device = i;
    }
  }
  
  // If this is stereo, then the autputBuffer is (L, R) tuples
  outputParameters.channelCount = 1; 
  // 32 bit floating point output
  outputParameters.sampleFormat = paFloat32;
  outputParameters.suggestedLatency = 
    Pa_GetDeviceInfo( outputParameters.device )->defaultLowOutputLatency;
  outputParameters.hostApiSpecificStreamInfo = NULL;

  retfail(Pa_OpenStream(
      &stream,
      NULL, // no input
      &outputParameters,
      SAMPLE_RATE, // sample rate
      AUDIO_BAND, // sample frames per buffer
      paNoFlag,
      audio_buffer_sync_callback,
      &audio_buf )); // this is context in the callback
  
  retfail(Pa_SetStreamFinishedCallback(stream, &cleanup));

  return SUCCESS;
}

void inplace_1d_convolve(
    float* source,
    int source_width,
    float* kernel,
    int kernel_width
    ) {
  float buff[source_width];
  for(int i = 0; i < source_width; ++i){
      buff[i] = 0.;
  }

  for(int i = 0; i < source_width; i++) {
    int k_min = fmax(i - kernel_width / 2, 0);
    int k_max = fmin(i + kernel_width / 2 + 1, source_width);
    for(int j = k_min; j < k_max; j++) {
      buff[i] += source[j] * kernel[j - k_min];
      if( source[j] > 0.01) {
        //printf("%d %d %d :: %f %f %f\n", i, j, j - k_min, buff[i], source[j], kernel[j - k_min]);
      }
    }
  }
  memmove(source, buff, source_width);
}

void display() {
  float (*input)[WIDTH][INPUT_DIM] = (void *) cppn[0].activations.e;

  for(int i=0; i < HEIGHT; ++i) {
    for(int j=0; j < WIDTH; ++j) {
      input[i][j][0] = (float) j / (WIDTH/2) -1.0;
      input[i][j][1] = (float) i / (HEIGHT/2) -1.0;
      input[i][j][2] = (float) frame_count / (SECONDS * FPS / 2) -1.0;
    }
  }

  matrix res = feedforward(cppn, num_layers);
  float (*output)[WIDTH][COLOURS] = 
    (void *) cppn[last_layer].activations.e;

  float harmonics[AUDIO_BAND];
  for(int i = 0; i < AUDIO_BAND; i++) harmonics[i] = 0.;
  float freqs[] = // key of A
   // A    B       C#      D       E       F#      G#
    { 440, 493.88, 554.37, 587.33, 659.25, 739.99, 830.61 };

  for(int note=0; note < sizeof(freqs) / sizeof(float); ++note) {
    float bin = (freqs[note]/2) / (SAMPLE_RATE/ (float) AUDIO_BAND);
    for(; (int) round(bin) < AUDIO_BAND; bin *= 2.0) {
      harmonics[(int) round(bin)] = 1.0;
    }
  }

  float kernel[5] = { 0.06136,	0.24477,	0.38774,	0.24477,	0.06136};
  inplace_1d_convolve(harmonics, AUDIO_BAND, kernel, 5);

  //for(int i = 0; i < AUDIO_BAND; i++) printf("%f\n", harmonics[i]);
  //exit(0);

  for(int i=0; i < HEIGHT; ++i) {
    for(int j=0; j < WIDTH; ++j) {
      output[i][j][0] *= harmonics[j];
      if(COLOURS == 3) {
        output[i][j][1] *= harmonics[j];
        output[i][j][2] *= harmonics[j];
      }
    }
  }
  render_buffer(res);

  // only print once per second
  clock_t curtime = clock();
  if ( curtime - lasttime >= CLOCKS_PER_SEC ){ 
    printf("FPS: %d\r", (frame_count - lastframe));
    fflush(stdout);
    lastframe = frame_count;
    lasttime = curtime;
  }
  
}

int shutdown() {
    retfail(Pa_StopStream( stream ));
    retfail(Pa_CloseStream( stream ));
    return SUCCESS;
}

void sighandler(int signo) {
  if (signo == SIGKILL || signo == SIGINT) {
    printf("Shutting down...");
    shutdown();
    exit(0);
  }
}

int keyboard_callback(unsigned char key, int x, int y) {
  printf("Keypress: %d\n", key);
  if(key == 'R') { // Reseed
    seed_network();
  } else if (key == 27) { // escape
    shutdown();
    exit(0);
  }
  return SUCCESS;
}

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
  srand(time(NULL));
  printf("Hello deepnet");
  retfail(init_neural_network());

  printf("Hello sound\n");
  retfail(init_portaudio());
  retfail(signal(SIGINT, sighandler) == SIG_ERR);
  retfail(signal(SIGKILL, sighandler) == SIG_ERR);
  retfail(Pa_StartStream(stream));
  
  printf("Hello video\n");
  retfail(init_display(argc, argv));
  
  glutDisplayFunc(&display);
  glutTimerFunc(0, &timer, 0);
  glutKeyboardFunc(&keyboard_callback);

  glutMainLoop(); // never returns
  return SUCCESS;
}
