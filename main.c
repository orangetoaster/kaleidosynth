#include <stdio.h>
#include <math.h>
#include <portaudio.h>

typedef int retcode;
#define retfail(CONDITION) { retcode __retval = CONDITION ; if((__retval < 0)) { fprintf(stderr, "Failed: %s\n In %s on line %d\n", #CONDITION, __FILE__, __LINE__ ); return __retval; } }

#define TABLE_SIZE 200
typedef struct {
  float buffer_data[TABLE_SIZE]; // pre-buffer size
  int left_phase;
  int right_phase;
} LRAudioBuf;

// This can be called at interrupt level, so nothing fancy, no malloc/free
static int audio_buffer_sync_callback(
    const void *inputBuffer, // unused
    void *outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo, // unused
    PaStreamCallbackFlags stats, // unused
    void *context ) {

  LRAudioBuf *audio_buf = (LRAudioBuf*)context;
  float *out = (float*)outputBuffer;
  unsigned long i;

  for( i=0; i<framesPerBuffer; i++ ) {
    *out++ = audio_buf->buffer_data[audio_buf->left_phase];  // left channel
    *out++ = audio_buf->buffer_data[audio_buf->right_phase];  // right channel
    audio_buf->left_phase += 1;
    if( audio_buf->left_phase >= TABLE_SIZE ) audio_buf->left_phase -= TABLE_SIZE;
    audio_buf->right_phase += 8;
    if( audio_buf->right_phase >= TABLE_SIZE ) audio_buf->right_phase -= TABLE_SIZE;
  }
  return paContinue;
}

static int init_portaudio(PaStream **stream) {
  PaStreamParameters outputParameters;
  LRAudioBuf audio_buf;
  int i;

  /* initialise sinusoidal wavetable */
  for( i=0; i<TABLE_SIZE; i++ ) {
    audio_buf.buffer_data[i] = (float) sin( ((double)i/(double)TABLE_SIZE) * M_PI * 2. );
  }
  audio_buf.left_phase = audio_buf.right_phase = 0;

  retfail(Pa_Initialize());

  outputParameters.device = Pa_GetDefaultOutputDevice(); /* default output device */
  retfail(outputParameters.device == paNoDevice) // no default output device
  
  outputParameters.channelCount = 2;       /* stereo output */
  outputParameters.sampleFormat = paFloat32; /* 32 bit floating point output */
  outputParameters.suggestedLatency = Pa_GetDeviceInfo( outputParameters.device )->defaultLowOutputLatency;
  outputParameters.hostApiSpecificStreamInfo = NULL;

  return Pa_OpenStream(
      *stream,
      NULL, // no input
      &outputParameters,
      44100, // sample rate
      64, // sample frames per buffer
      paClipOff,
      audio_buffer_sync_callback,
      &audio_buf ); // this is context in the callback
}

static int play_sound(PaStream **stream, float seconds) {
    retfail(Pa_StartStream(*stream));

    Pa_Sleep(seconds * 1000 );

    retfail(Pa_StopStream( *stream ));
    retfail(Pa_CloseStream( *stream ));
    return Pa_Terminate();
}

int main() {
  PaStream *stream;
  printf("Hello sound\n");
  retfail(init_portaudio(&stream));
  retfail(play_sound(stream, .5));
  printf("All good.\n");
  return 0;
}
