#ifndef ERR_H
#define ERR_H
typedef int FD;
typedef unsigned char uchar;
typedef int retcode;
static const retcode SUCCESS = 0;
static const retcode FAIL = -1;

typedef int retcode;
#define retfail(CONDITION) { retcode __retval = CONDITION ; if((__retval < 0)) \
  { fprintf(stderr, "Failed: %s\n In %s on line %d - %d\n", #CONDITION, __FILE__, __LINE__, __retval ); return __retval; } }


#endif
