/*
 * $Id: dbglog.h 25 2009-03-26 16:15:16Z burlog $
 *
 * DBGLOG -- debug and runtime logger
 *
 * AUTHOR: Vaclav Blazek <blazek@firma.seznam.cz>
 *
 */

#include <stdio.h>
#include <sys/types.h>

#ifndef __DBGLOG_H__
#define __DBGLOG_H__

#ifndef __GNUC__
#error "Oops! You must use GCC to compile this!"
#endif

#define INFO1  0x0000000f
#define INFO2  0x00000007
#define INFO3  0x00000003
#define INFO4  0x00000001

#define WARN1  0x000000f0
#define WARN2  0x00000070
#define WARN3  0x00000030
#define WARN4  0x00000010

#define ERR1   0x00000f00
#define ERR2   0x00000700
#define ERR3   0x00000300
#define ERR4   0x00000100

#define FATAL1 0x0000f000
#define FATAL2 0x00007000
#define FATAL3 0x00003000
#define FATAL4 0x00001000

#define DBG1   0x0f000000
#define DBG2   0x07000000
#define DBG3   0x03000000
#define DBG4   0x01000000

#define DBG_INTERNAL 0xffffffff

#define PID_APPNAME ((char*)(-1))
#define PID_TID_APPNAME ((char*)(-2))

#define LOG(level, format...) \
  __dbglog(level, (const char*) __FILE__, (const char*)__FUNCTION__, \
       __LINE__, format)

#define LOGF(level, file, format...) \
  __dbglogf(level, file, (const char*) __FILE__, (const char*)__FUNCTION__, \
        __LINE__, format)

#ifdef DEBUG
#define DBG(level, format...) \
  __dbglog(level, (const char*) __FILE__, (const char*)__FUNCTION__, \
       __LINE__, format)

#define DBGF(level, file, format...) \
  __dbglogf(level, file, (const char*) __FILE__, (const char*)__FUNCTION__, \
        __LINE__, format)
#else
#define DBG(level, format...) do {} while(0)
#define DBGF(level, file, format...) do {} while(0)
#endif

#ifdef __cplusplus
extern "C" {
#endif
    typedef int (*dbglogFormatter_t)(char *buff, int size, const void *format,
                                     void *data, void *additionalData);

    int __dbglog(unsigned level, const char *source, const char *func,
                 size_t line, const char *format, ...)
        __attribute__ ((format (printf, 5, 6)));
    
    int __dbglogf(unsigned level, FILE *file, const char *source,
                  const char *func, size_t line, const char *format, ...)
        __attribute__ ((format (printf, 6, 7)));
    
    int dbglog(unsigned level, FILE *file, const char *source,
               const char *func, size_t line, const void *format,
               void *data, dbglogFormatter_t formatter,
               void *formatterData);
        
    void logMask(const char *mask);
    int logFile(const char *file);
    int logReopen();
    void logAppName(const char *name);
    const char* logGetAppName();
    void logStderr(int err);
    int logStderrValue();
    void logInit(void);
    void logMaskSource(unsigned *maskSource);
    void logMaskValue(char *mask, size_t len);
    int logOwnerName(const char *user, const char *group);
    int logOwnerId(uid_t user, gid_t group);
    void logBufSize(unsigned bufSize);
    void logUseStderr(int use);
    int logCheckLevel(int level);
    void logTimePrecision(unsigned int precision);
    void logTimePrecisionSource(unsigned int *precisionSource);
    unsigned int logTimePrecisionValue();
    const char* logFileValue();

    void logReinitialize();
#ifdef __cplusplus
}
#endif

#endif // __DBGLOG_H__
