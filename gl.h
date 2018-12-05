#define GL_SILENCE_DEPRECATION

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glut.h>
#include <GL/glu.h>
#endif

#include "nn.h"

#define FPS 12
#define WIDTH 320
#define HEIGHT 240
#define COLOURS 3
#define INPUT_DIM 3

/// VIDEO GLOBALS ///
static volatile int frame_count = 0;
static volatile int lastframe = 0;
static const int SECONDS = 3;
static volatile int SHIFT_COLOURS = 0;
static int framebuffer_id = 0;

void create_framebuffer() {
  glGenTextures(1, &framebuffer_id);
  glBindTexture(GL_TEXTURE_2D, framebuffer_id);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
}

void render_buffer(float *buffer) {
  if(COLOURS == 1) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, 
      WIDTH, HEIGHT, 0, GL_LUMINANCE, GL_FLOAT, buffer);
  } else if (COLOURS == 3) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB,
               GL_FLOAT, buffer - SHIFT_COLOURS * sizeof(float));
  }
  //glPolygonMode(GL_FRONT_AND_BACK, self->polygon_mode);
  
  glBegin(GL_QUADS);
    glTexCoord2f( 1.0f, 1.0f);
    glVertex3f  ( 1.0f, 1.0f,0.0f);
    
    glTexCoord2f( 0.0f, 1.0f);
    glVertex3f  (-1.0f, 1.0f,0);
    
    glTexCoord2f( 0.0f, 0.0f);
    glVertex3f  (-1.0f,-1.0f,0);
    
    glTexCoord2f( 1.0f, 0.0f);
    glVertex3f  ( 1.0f,-1.0f,0);
  glEnd();

  glutSwapBuffers(); /* calls glFlush() */
}


int init_display(int argc, char **argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
  glutInitWindowSize((int)WIDTH*2.5, (int) HEIGHT*2.5);

  glutCreateWindow("Kaleidosynth");
  //glutFullScreen();
 
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_TEXTURE_2D);
  
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

  create_framebuffer();

  return SUCCESS;
}

