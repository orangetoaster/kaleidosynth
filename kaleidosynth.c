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
#include <limits.h>
#include <fftw3.h>

float framebuffer_unsnake[HEIGHT][WIDTH][COLOURS];
/// AUDIO GLOBALS ///  
static const float gaussian_kernel[] = 
    { 0.006, 0.06136, 0.24477, 0.38774, 0.24477, 0.06136, 0.006};
static const int glen= sizeof(gaussian_kernel) / sizeof(float);
static volatile int CLAMP_KEY = INT_MAX;
#define AUDIO_FACTOR 4
#define AUDIO_BAND (WIDTH *HEIGHT * COLOURS)
typedef struct {
/*  kiss_fft_scalar buffer_data[AUDIO_BAND]; // pre-buffer size
  kiss_fft_scalar copy_buf[AUDIO_BAND]; // pre-buffer size
  kiss_fft_scalar freq_data[AUDIO_BAND]; // pre-buffer size
  kiss_fft_scalar melody[AUDIO_BAND]; // pre-buffer size*/
  int left_phase;
  int note_pos;
  int maxpos;
  int colourphase;
  kiss_fftr_cfg cfg;
} LRAudioBuf;
LRAudioBuf audio_buf = { 0 };
PaStream *stream = NULL;
static const float volumeMultiplier = 1.0; //0.01f;
static const float SAMPLE_RATE = 44100;

static volatile clock_t lasttime = 0;
float harmonics[AUDIO_BAND];
float frequency_space[AUDIO_BAND];
float audio_double_buf[AUDIO_BAND][COLOURS];
kiss_fftr_cfg full_fftri_cfg = {0};
kiss_fftr_cfg full_fftr_cfg = {0};

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
    }
  }
  memmove(source, buff, source_width * sizeof(float));
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

  for(unsigned long i=0; i<framesPerBuffer; i = (i+1) % (WIDTH*HEIGHT)) {
    float (*nn_l)[COLOURS] = (void *) 
      cppn[last_layer].activations.e;
    //*out++ = audio_buf->buffer_data[audio_buf->left_phase]
    //*out++ = nn_l[audio_buf->left_phase++][audio_buf->colourphase] //+ nn_l[i][1] + nn_l[i][2])
    // left
    *out++ = audio_double_buf[audio_buf->left_phase][audio_buf->colourphase] //+ nn_l[i][1] + nn_l[i][2])
      * volumeMultiplier;
    // right
    *out++ = audio_double_buf[audio_buf->left_phase][(audio_buf->colourphase + 1) % COLOURS] //+ nn_l[i][1] + nn_l[i][2])
      * volumeMultiplier;
    audio_buf->left_phase ++;
    if(audio_buf->left_phase >= WIDTH*HEIGHT) {
      audio_buf->left_phase -= WIDTH*HEIGHT;
      audio_buf->colourphase = (audio_buf->colourphase + 1) % 3;
      if(audio_buf->colourphase ==2) {
        if(SHIFT_COLOURS == 0) {
          SHIFT_COLOURS = 1;
        } else {
          SHIFT_COLOURS = 0;
        }
      }
    }

/*    audio_buf->left_phase += 1;
    // refill the audio buffer with the ifft of the visualization scanning
    // down the screen
    if( audio_buf->left_phase >= AUDIO_BAND ) {
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
      
      for (int j=0; j < AUDIO_FACTOR; ++j) {
        for (int k=0; k < WIDTH; ++k) {
          if ( (p +j) % 2 == 0) {
            audio_buf->freq_data[j*WIDTH + k] = nn_l[p + j][k][0];
          } else {
            audio_buf->freq_data[j*WIDTH + k] = nn_l[p + j][WIDTH - (k + 1)][0];
          }
        /*if(COLOURS == 3) {
          audio_buf->freq_data[j] += nn_l[p][j/AUDIO_FACTOR][1] 
            + nn_l[p][j/AUDIO_FACTOR][2];
        }
        }
      }
/*      // scan for the max entry and play that note
      for (int j=0; j < AUDIO_BAND; ++j) { 
        if (audio_buf->freq_data[j] > audio_buf->freq_data[audio_buf->maxpos] &&
            abs(audio_buf->freq_data[j]) > 0.09 // Only if it's loud enough to change
           ) {
          audio_buf->melody[audio_buf->maxpos] = 0;
          audio_buf->maxpos = j;
          audio_buf->melody[audio_buf->maxpos] =
            audio_buf->freq_data[j];*/

      /*float smooth[AUDIO_FACTOR];
      for(int a=0; a < AUDIO_FACTOR; ++a) {
        smooth[a] = 1.0 / AUDIO_FACTOR;
      }
      inplace_1d_convolve(audio_buf->freq_data, (int) AUDIO_BAND, 
        smooth, AUDIO_FACTOR);

      float fft_output[AUDIO_BAND];
      kiss_fftri(audio_buf->cfg, 
        (kiss_fft_cpx *) &audio_buf->freq_data,
        audio_buf->buffer_data);

      /*  for (int i=0; i < AUDIO_BAND; ++i) {
          audio_buf->buffer_data[i] = fft_output[i] * 0.5 
            + audio_buf->buffer_data[i] * 0.5;
        }
    }
      */
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
//  audio_buf.cfg = kiss_fftr_alloc(AUDIO_BAND, 1, NULL,NULL);

  retfail(Pa_Initialize());

  /* default output device */
  outputParameters.device = Pa_GetDefaultOutputDevice(); 
  retfail(outputParameters.device == paNoDevice) 
  int num_devs = Pa_GetDeviceCount();
  PaDeviceInfo *di = NULL;
  for (int i=0; i < num_devs ; ++ i) {
    di = Pa_GetDeviceInfo(i);
    printf("Available Adev: %s %lf\n", di->name, di->defaultSampleRate);
    if(strcmp(di->name, "pulse") == 0) {
      printf("Choosing this device\n");
      chosen_device = i;
    }
  }
  
  // If this is stereo, then the autputBuffer is (L, R) tuples
  outputParameters.channelCount = 2; 
  // 32 bit floating point output
  outputParameters.sampleFormat = paFloat32;
  outputParameters.suggestedLatency = 
    Pa_GetDeviceInfo( outputParameters.device )->defaultLowOutputLatency;
  outputParameters.hostApiSpecificStreamInfo = NULL;

  retfail(Pa_OpenStream(
      &stream,
      NULL, // no input
      &outputParameters,
      44100.0, //SAMPLE_RATE, // sample rate
      paFramesPerBufferUnspecified, // sample frames per buffer
      paNoFlag,
      audio_buffer_sync_callback,
      &audio_buf )); // this is context in the callback
  
  retfail(Pa_SetStreamFinishedCallback(stream, &cleanup));
  
  // setup key structures //
  const float BANDPASS = 15000. / (SAMPLE_RATE / (float) AUDIO_BAND);
  for(int i = 0; i < AUDIO_BAND; i++) harmonics[i] = 0.;
  float freqs[] = // key of A
   // A    B       C#      D       E       F#      G#
    { 440, 493.88, 554.37, 587.33, 659.25, 739.99, 830.61 };
  for(int note=0; note < sizeof(freqs) / sizeof(float); ++note) {
    int bin = (int) round((freqs[note]/2.0) 
                / (SAMPLE_RATE/ (float) AUDIO_BAND));
    float falloff = 1.0;
    for(; bin < AUDIO_BAND; bin *= 4.0) { // double for octave and phase
      harmonics[bin] = 1.0 * falloff;
      falloff /= 3.0;
    }
  }

  float sq_gaussian_kernel[glen*2];
  memset(sq_gaussian_kernel, 0.0, glen*2);
  for( int i =0; i < glen; ++i) {
    sq_gaussian_kernel[i*2] = sqrt(sqrt(gaussian_kernel[i]));
  }
  inplace_1d_convolve(harmonics, (int) AUDIO_BAND, sq_gaussian_kernel, glen);


  return SUCCESS;
}

void display() {
  float (*input)[WIDTH][INPUT_DIM] = (void *) cppn[0].activations.e;

  for(int i=0; i < HEIGHT; ++i) {
    for(int j=0; j < WIDTH; ++j) {
      if(i %2 == 0) {
        input[i][j][0] = (float) j / (WIDTH/2) -1.0;
        input[i][j][1] = (float) i / (HEIGHT/2) -1.0;
      } else {
        input[i][j][0] = (float) (WIDTH - (j+1)) / (WIDTH/2) -1.0;
        input[i][j][1] = (float) i / (HEIGHT/2) -1.0;
      }
      input[i][j][2] = (float) frame_count / (SECONDS * FPS / 2) -1.0;
    }
  }

  matrix res = feedforward(cppn, num_layers);
  
  float (*output)[WIDTH][COLOURS] = 
    (void *) cppn[last_layer].activations.e;
 
//  fftw_real /*in[AUDIO_BAND], out[AUDIO_BAND],*/ power_spectrum[AUDIO_BAND/2+1];
//  rfftw_plan p;
//  int k;
//  p = rfftw_create_plan(AUDIO_BAND, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE);
//  rfftw_one(p, output, frequency_space);
//  power_spectrum[0] = frequency_space[0]*frequency_space[0];  /* DC component */
//  /* (k < AUDIO_BAND/2 rounded up) */
//  for (k = 1; k < (AUDIO_BAND+1)/2; ++k) {
//    power_spectrum[k] = frequency_space[k]*frequency_space[k] + frequency_space[AUDIO_BAND-k]*frequency_space[AUDIO_BAND-k];
//  }
//  if (AUDIO_BAND % 2 == 0) {/* AUDIO_BAND is even */
//    /* Nyquist freq. */
//    power_spectrum[AUDIO_BAND/2] = frequency_space[AUDIO_BAND/2]*frequency_space[AUDIO_BAND/2];
//  }
//  rfftw_destroy_plan(p);
//
  kiss_fftr(full_fftr_cfg, 
      output,
      (kiss_fft_cpx *) frequency_space);

  if(CLAMP_KEY != INT_MAX) {
    for(int i=0; i < AUDIO_BAND; i+=2) {
      frequency_space[i] *= harmonics[i];
    }
  }
  
  kiss_fftri(full_fftri_cfg, 
      (kiss_fft_cpx *) frequency_space,
      output);
  for(int i=0; i < AUDIO_BAND * COLOURS; ++i) { // tada, unnormalized output
    cppn[last_layer].activations.e[i] /= (AUDIO_BAND/2);
  }
  memcpy(audio_double_buf, output, AUDIO_BAND*COLOURS);

  // unsnake what gets rendered, or it's super abstract and doesn't look cppn
  for(int i=0; i < HEIGHT; ++i) {
    for(int j=0; j < WIDTH; ++j) {
      for(int k=0; k < COLOURS; ++k) {
        if(i %2 == 0) {
          framebuffer_unsnake[i][j][k] = output[i][j][k];
        } else {
          framebuffer_unsnake[i][WIDTH - (j+1)][k] = output[i][j][k];
        }
      }
    }
  }
  
  render_buffer((float *) &framebuffer_unsnake);

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
  } else if (key == 'a') {
    CLAMP_KEY = 0;
  } else if (key == 'b') {
    CLAMP_KEY = 1;
  } else if (key == 'c') {
    CLAMP_KEY = 2;
  } else if (key == 'd') {
    CLAMP_KEY = 3;
  } else if (key == 'e') {
    CLAMP_KEY = 4;
  } else if (key == 'f') {
    CLAMP_KEY = 5;
  } else if (key == 'g') {
    CLAMP_KEY = 6;
  } else if (key == ' ') {
    CLAMP_KEY = INT_MAX;
  }
  return SUCCESS;
}

void timer(int value) {
  glutPostRedisplay();
  glutTimerFunc(1000 / FPS, &timer, value);
  frame_count ++;
  if(frame_count > 60 * SECONDS) {
    frame_count = 0;
    seed_network();
  }
}

int main(int argc, char **argv) {
  srand(time(NULL));
  printf("Hello deepnet");
  retfail(init_neural_network());
  full_fftri_cfg = kiss_fftr_alloc(WIDTH * HEIGHT * COLOURS, 1, NULL,NULL);
  full_fftr_cfg = kiss_fftr_alloc(WIDTH * HEIGHT * COLOURS, 0, NULL,NULL);

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
