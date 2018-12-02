// So, we should have a quad that has a texture buffer we can double-buffer to and update realtime.

int model_draw_next(model self) {
  glEnable(GL_TEXTURE_2D);
  int ctrl_i = 0;

  for (ctrl_i = 0; ctrl_i < 21; ctrl_i ++) {
    if (self->curframe <= anim_ctrl[ctrl_i][1]) {
      self->tween += (double) anim_ctrl[ctrl_i][2] / self->global_fps;
      break;
    }
  }

  if (self->tween >= 1) { // keyframe advance //
    self->tween = 0;
    self->curframe = (self->curframe + 1) % (self->num_frames - 1);
  }

  glPolygonMode(GL_FRONT_AND_BACK, self->polygon_mode);
  glBindTexture(GL_TEXTURE_2D, self->tex_id); // Select correct texture //
  int retval = self->draw_routine(self);
  glDisable(GL_TEXTURE_2D);
  return retval;
}

static int model_load_tex(model self, char *filename) {
  PCX pcx_tex = new (PCX_class);                // Try pcx //

  if (PCX_Load(pcx_tex, filename) == 0) {
    glGenTextures(1, &self->tex_id);
    glBindTexture(GL_TEXTURE_2D, self->tex_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, PCX_numColours(pcx_tex), PCX_width(pcx_tex), PCX_height(pcx_tex), 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, PCX_img(pcx_tex));
  }

  delete (pcx_tex);
  BMP bmp_tex = new (BMP_class);                // Try bmp //

  if (BMP_Load(bmp_tex, filename) == 0) {
    glGenTextures(1, &self->tex_id);
    glBindTexture(GL_TEXTURE_2D, self->tex_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, BMP_numColours(bmp_tex), BMP_width(bmp_tex), BMP_height(bmp_tex), 0, GL_BGR,
                 GL_UNSIGNED_BYTE, BMP_img(bmp_tex));
  }

  delete (bmp_tex);
  return 1;
}

