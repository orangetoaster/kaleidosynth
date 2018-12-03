#include <stdio.h>
#include "nn.h"
#include <math.h>
#include <portaudio.h>
#include <complex.h>
#include "kiss_fftr.h"

typedef int retcode;
#define retfail(CONDITION) { retcode __retval = CONDITION ; if((__retval < 0)) \
  { fprintf(stderr, "Failed: %s\n In %s on line %d - %d\n", #CONDITION, __FILE__, __LINE__, __retval ); return __retval; } }

#define TABLE_SIZE 2048
typedef struct {
  kiss_fft_scalar buffer_data[TABLE_SIZE]; // pre-buffer size
  kiss_fft_cpx freq_domain[TABLE_SIZE/2]; // pre-buffer size
  int left_phase;
  int note_pos;
  kiss_fftr_cfg cfg;
} LRAudioBuf;

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
      audio_buf->freq_domain[audio_buf->note_pos].i = 0;
      audio_buf->note_pos = (audio_buf->note_pos + 1) % (TABLE_SIZE/2);
      audio_buf->freq_domain[audio_buf->note_pos].i = -200;
      kiss_fftri(audio_buf->cfg, audio_buf->freq_domain, audio_buf->buffer_data);
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

LRAudioBuf audio_buf = { 0 };
static int init_portaudio(PaStream **stream) {
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
      stream,
      NULL, // no input
      &outputParameters,
      44100, // sample rate
      64, // sample frames per buffer
      paClipOff,
      audio_buffer_sync_callback,
      &audio_buf )); // this is context in the callback
  
  retfail(Pa_SetStreamFinishedCallback(*stream, &cleanup));

  return 0;
}

static int play_sound(PaStream **stream, float seconds) {
    retfail(Pa_StartStream(stream));

    Pa_Sleep(seconds * 1000 );

    retfail(Pa_StopStream( stream ));
    retfail(Pa_CloseStream( stream ));
    return 0;
}

static int neural_network() {
  static const int pixcount = 10;
  static const int hidden_neurons = 30, output_neurons = 10;
  static const float initialization_sigma = 0.50;
  static const int epochs = 10;
  float learning_rate = 0.01f, decay = 1.001f;
  struct neural_layer cppn[] = {
    {
      .weights = { 0 }, .w_delt = { 0 }, .biases = { 0 }, .b_delt = { 0 },
      .activations = { .x = 1, .y = pixcount, .e = malloc(pixcount * sizeof(float)) },
      .activate = NULL, .backprop = NULL,
    }, {
      .weights = { .x = pixcount, .y = hidden_neurons, .e = malloc(pixcount *hidden_neurons * sizeof(float)) },
      .w_delt = { .x = pixcount, .y = hidden_neurons, .e = calloc(pixcount * hidden_neurons, sizeof(float)) },
      .biases = { .x = 1, .y = hidden_neurons, .e = malloc(hidden_neurons * sizeof(float)) },
      .b_delt = { .x = 1, .y = hidden_neurons, .e = calloc(hidden_neurons, sizeof(float)) },
      .activations = { .x = 1, .y = hidden_neurons, .e = calloc(hidden_neurons, sizeof(float)) },
      .activate = &sigmoid_activate,
      .backprop = &sigmoid_deriv,
    }, {
      .weights = { .x = hidden_neurons, .y = output_neurons, .e = malloc(hidden_neurons *output_neurons * sizeof(float)) },
      .w_delt = { .x = hidden_neurons, .y = output_neurons, .e = calloc(hidden_neurons * output_neurons, sizeof(float)) },
      .biases = { .x = 1, .y = output_neurons, .e = malloc(output_neurons * sizeof(float)) },
      .b_delt = { .x = 1, .y = output_neurons, .e = calloc(output_neurons, sizeof(float)) },
      .activations = { .x = 1, .y = output_neurons, .e = calloc(output_neurons, sizeof(float)) },
      .activate = &sigmoid_activate,
      .backprop = &sigmoid_deriv,
    },
  };

  static const int num_layers = 3;

  for (int i = 1; i < num_layers; ++i) {
    randomize(cppn[i].weights.e, cppn[i].weights.x * cppn[i].weights.y, initialization_sigma);
    randomize(cppn[i].biases.e, cppn[i].biases.y, initialization_sigma);
  }

  matrix res = feedforward(cppn, num_layers);
  printf("%f", res.e[0]);
}

int main() {
  PaStream *stream = NULL;
  printf("Hello sound\n");
  retfail(init_portaudio(&stream));
  retfail(play_sound(stream, 30));
  Pa_Terminate();

  printf("All good.\n");
  return 0;
}
