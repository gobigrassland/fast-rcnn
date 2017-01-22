/*
 * $Id: dbglog.cc 35 2011-05-31 10:27:16Z volca $
 *
 * DBGLOG -- debug and runtime logger
 *
 * AUTHOR: Vaclav Blazek <blazek@firma.seznam.cz>
 *
 */

#ifdef HAVE_GETTID
#include <unistd.h>
#include <sys/syscall.h>
#endif

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <pwd.h>
#include <grp.h>

#include <errno.h>
#include <sys/resource.h>

#include "caffe/3rdparty/dbglogobject.h"

DbgLog_t *DbgLog_t::defLog = 0;

const pthread_mutex_t DbgLog_t::initializer = PTHREAD_MUTEX_INITIALIZER;

void DbgLog_t::initializeDefLog() {
    if (!DbgLog_t::defLog) {
        // create and initialize default log
        DbgLog_t::defLog = new DbgLog_t();
        DbgLog_t::defLog->init();
    }

    // Signal SIGXFSZ se ignoruje aby nam nedelal neplechu. Normalne
    // by nemel byt procesu zaslan, protoze je dbglog kompiloval pro
    // LFS (64 bit FS). Pokud nekdo omezi velikost souboru pres ulimit
    // (setrlimit), muze byt tento signal dorucen.
    struct sigaction sigact;
    sigact.sa_handler = SIG_IGN;
    sigemptyset(&sigact.sa_mask);
    sigact.sa_flags = SA_RESETHAND | SA_RESTART;
    sigaction(SIGXFSZ, &sigact, 0);
}

// just initialize DbgLog_t::defLog
class StaticInitializer_t {
private:
    StaticInitializer_t() {
        DbgLog_t::initializeDefLog();
    }

    // static member, will call constructor => will call
    // DbgLog_t::initializeDefLog();
    static StaticInitializer_t dbglogStaticinitializer;
};

// static initializer
StaticInitializer_t StaticInitializer_t::dbglogStaticinitializer;

void logReinitialize() {
    DbgLog_t::initializeDefLog();
}

int __dbglog(unsigned level, const char *source, const char *func,
             size_t line, const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    int ret = DbgLog_t::defLog->log(level, NULL, source,
                                    func, line, format, &ap);
    va_end(ap);
    return ret;
}

int __dbglogf(unsigned level, FILE *file, const char *source,
              const char *func, size_t line, const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    int ret = DbgLog_t::defLog->log(level, file, source,
                                    func, line, format, &ap);
    va_end(ap);
    return ret;
}

int dbglog(unsigned level, FILE *file, const char *source,
           const char *func, size_t line, const void *format,
           void *data, dbglogFormatter_t formatter,
           void *formatterData)
{
    return DbgLog_t::defLog->log(level, file, source, func, line,
                                 format, data, formatter,
                                 formatterData);
}

void logMask(const char *mask) {
    DbgLog_t::defLog->setMask(mask);
}

int logFile(const char *file) {
    return DbgLog_t::defLog->setFile(file);
}

int logReopen() {
    return DbgLog_t::defLog->reopenFile();
}

void logAppName(const char *name) {
    DbgLog_t::defLog->setAppName(name);
}

const char* logGetAppName() {
    return DbgLog_t::defLog->getAppName();
}

void logStderr(int err) {
    DbgLog_t::defLog->useStderr(err);
}

int logStderrValue() {
    return DbgLog_t::defLog->getUseStderr();
}

void logInit(void) {
    // explicit call of auto-initialization procedure
    DbgLog_t::defLog->init();
}

void logMaskSource(unsigned *maskSource) {
    DbgLog_t::defLog->setMaskSource(maskSource);
}

void logMaskValue(char *logMask, size_t len) {
    DbgLog_t::defLog->getMaskValue(logMask, len);
}

int logOwnerName(const char *user, const char *group) {
    return DbgLog_t::defLog->setOwner(user, group);
}

int logOwnerId(uid_t user, gid_t group) {
    return DbgLog_t::defLog->setOwner(user, group);
}

void logBufSize(unsigned bufSize) {
    DbgLog_t::defLog->setBufSize(bufSize);
}

void logUseStderr(int use) {
    DbgLog_t::defLog->bindStderr(use);
}

int logCheckLevel(int level) {
    return DbgLog_t::defLog->checkLevel(level);
}

void logTimePrecision(unsigned int precision) {
    DbgLog_t::defLog->setTimePrecision(precision);
}

void logTimePrecisionSource(unsigned int *precisionSource) {
    DbgLog_t::defLog->setTimePrecisionSource(precisionSource);
}

unsigned int logTimePrecisionValue() {
    return DbgLog_t::defLog->getTimePrecision();
}

const char* logFileValue() {
    return DbgLog_t::defLog->getLogFile();
}

// ---------------------------------------------------------------------------

int DbgLog_t::setFile(const char *fileName) {
    // mark that we should reopen (or just open) the file
    reopen = true;

    if (fileName) {
        ThreadLocker_t tlock(&locker);
        if (curFileName) free(curFileName);
        curFileName = strdup(fileName);
    }

    // always ok
    return 0;
}

int DbgLog_t::openFile() {
    if (!reopen) return 0;
    reopen = false;

    if (logFile >= 0) {
        close(logFile);
        logFile = -1;
    }

    errno = ENOENT;
    logFile = open(curFileName? curFileName: "/dev/null",
                   O_CREAT | O_WRONLY | O_APPEND,
                   S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP |
                   S_IROTH | S_IWOTH);

    // mark log file to be closed on exec
    if (logFile >= 0) ::fcntl(logFile, F_SETFD, FD_CLOEXEC);

    if (ownsStderr) {
        // we own stderr so we have to update it
        if (logFile >= 0) {
            // duplicate log file
            dup2(logFile, STDERR_FILENO);
        } else {
            // no log file => loose stderr
            int f = open("/dev/null", O_WRONLY | O_APPEND);
            dup2(f, STDERR_FILENO);
            close(f);
        }
    }

    // return status
    return -(logFile < 0);
}

void DbgLog_t::setMask(const char *maskStr) {
    ThreadLocker_t tlock(&locker);
    if (!maskStr || !strcmp(maskStr, "DEFAULT")) {
        // use default mask
        mask = ~(INFO3 | WARN2 | ERR2 | FATAL1);
        return;
    }

    switch (toupper(*maskStr)) {
    case 'A':
        mask = MASK_ALL;
        if (maskSource) *maskSource = mask;
        return;
    case 'N':
        mask = MASK_NONE;
        if (maskSource) *maskSource = mask;
        return;
    }

    int m = 0;

    for (;;) {
        char mode = *maskStr++;
        if (!mode) goto RETURN;
        char level = *maskStr++;
        if (!level) goto RETURN;

        unsigned value = 0;

        switch (level) {
        case '1': value = 0xf; break;
        case '2': value = 0x7; break;
        case '3': value = 0x3; break;
        case '4': value = 0x1; break;
        }

        switch (toupper(mode)) {
        case 'I': break;
        case 'W': value <<= 4; break;
        case 'E': value <<= 8; break;
        case 'F': value <<= 12; break;
        case 'D': value <<= 24; break;
        default: value = 0; break;
        }

        m |= value;
    }
 RETURN:
    mask = ~m;
    if (maskSource) *maskSource = mask;
}

static const unsigned MASK_ANY1  = 0x88888888;
static const unsigned MASK_ANY2  = 0x44444444;
static const unsigned MASK_ANY3  = 0x22222222;
static const unsigned MASK_ANY4  = 0x11111111;

struct FileLocker_t {
    FileLocker_t(int file)
        : fd(file)
    {
        lock();
    }


    FileLocker_t(FILE *file)
        : fd(fileno(file))
    {
        lock();
    }

    ~FileLocker_t() {
        struct flock l = {
            F_UNLCK, SEEK_SET, 0, 0, 0
        };

        int stat;
        do {
            stat = fcntl(fd, F_SETLKW, &l);
            /* following code may cause problems with
             * stopLogging feature (function may write()). */
            /*if (stat < 0) {
              char buf[128];
              snprintf(buf, sizeof(buf),
              "DBG-LOG UNLOCK ERROR (%s)\n", strerror(errno));
              write(fd, buf, strlen(buf));
              }*/
        }
        while (stat < 0 && errno == EINTR);
    }

private:
    inline void lock() {
        struct flock l = {
            F_WRLCK, SEEK_SET, 0, 0, 0
        };

        int stat;
        do {
            stat = fcntl(fd, F_SETLKW, &l);
            /* following code may cause problems with
             * stopLogging feature (function may write()). */
            /*if (stat < 0) {
              char buf[128];
              snprintf(buf, sizeof(buf),
              "DBG-LOG LOCK ERROR (%s)\n", strerror(errno));
              write(fd, buf, strlen(buf));
              }*/
        }
        while (stat < 0 && errno == EINTR);
    }

    int fd;
};

static inline void levelString(char *str, unsigned level) {
    // check for internal error
    if (level == DBG_INTERNAL) {
        str[0] = '!';
        str[1] = '!';
        str[2] = '\0';
    } else {
        // print severity
        *str++ = (level & INFO1) ? 'I' :
            (level & WARN1) ? 'W' :
            (level & ERR1) ? 'E' :
            (level & FATAL1) ? 'F' : 'D';

        // print level
        *str++ = (level & MASK_ANY1) ? '1' :
            (level & MASK_ANY2) ? '2' :
            (level & MASK_ANY3) ? '3' : '4';

        // terminate
        *str = '\0';
    }
}

static int makeLog(char *buff, ssize_t space,
                   const char *message, int messageLen,
                   const char *name, unsigned level,
                   unsigned int precision)
{
    // create level string
    char ls[3];
    levelString(ls, level);

    // get current time
    struct timeval tt;
    struct timezone tz;
    gettimeofday(&tt, &tz);

    time_t clock = tt.tv_sec;
    struct tm *t = localtime(&clock);

    // print time
    ssize_t total = 0;
    ssize_t written = 0;

    switch (precision) {
    case 6:
        // full microseconds
        written = snprintf(buff, space, "%04d/%02d/%02d %02d:%02d:%02d.%06ld ",
                           t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                           t->tm_hour, t->tm_min, t->tm_sec,
                           tt.tv_usec);
        break;
    default:
        // any other value => normal log
        written = snprintf(buff, space, "%04d/%02d/%02d %02d:%02d:%02d ",
                           t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                           t->tm_hour, t->tm_min, t->tm_sec);
        break;
    }

    // check for error
    if ((written < 0) || (written >= space)) {
        memcpy(buff + space - 4, "...\n", 4);
        return total + space;
    }
    // advance
    buff += written; space -= written; total += written;

    // print severity && level string
    written = snprintf(buff, space, "%s: ", ls);
    // check for error
    if ((written < 0) || (written >= space)) {
        memcpy(buff + space - 4, "...\n", 4);
        return total + space;
    }
    // advance
    buff += written; space -= written; total += written;

    // print app name
    if (name) {
        if (name == PID_APPNAME) {
            // print [pid] as app name
            written = snprintf(buff, space, "[%d]: ", int(getpid()));
            // check for error
            if ((written < 0) || (written >= space)) {
                memcpy(buff + space - 4, "...\n", 4);
                return total + space;
            }
            // advance
            buff += written; space -= written; total += written;
        } else if (name == PID_TID_APPNAME) {
            // print [pid:tid] as app name
#if SIZEOF_PTHREAD_T == 8
# ifdef HAVE_GETTID
            written = snprintf(buff, space, "[%d:%lu:(%d)]: ", int(getpid()),
                               pthread_self(), int(syscall(SYS_gettid)));
# else
            written = snprintf(buff, space, "[%d:%lu]: ", int(getpid()),
                               pthread_self());
# endif
#elif SIZEOF_PTHREAD_T == 4
# ifdef HAVE_GETTID
            written = snprintf(buff, space, "[%d:%u:(%d)]: ", int(getpid()),
                               pthread_self(), int(syscall(SYS_gettid)));
# else
            written = snprintf(buff, space, "[%d:%u]: ", int(getpid()),
                               pthread_self());
# endif
#endif
            // check for error
            if ((written < 0) || (written >= space)) {
                memcpy(buff + space - 4, "...\n", 4);
                return total + space;
            }
            // advance
            buff += written; space -= written; total += written;
        } else {
            // if app name is nonempty, print it
            if (*name) {
                written = snprintf(buff, space, "%s: ", name);
                // check for error
                if ((written < 0) || (written >= space)) {
                    memcpy(buff + space - 4, "...\n", 4);
                    return total + space;
                }
                // advance
                buff += written; space -= written; total += written;
            }
        }
    }

    if (messageLen >= space) {
        memcpy(buff, message, space);
        return total + space;
    }
    else {
        memcpy(buff, message, messageLen);
        return total + messageLen;
    }
}

static int makeLogMessage(char *buff, ssize_t space,
                          const void *format, void *data,
                          dbglogFormatter_t formatter,
                          void *formatterData)
{
    int written = 0;

    // format data
    if (formatter) {
        // call formatter
        written = formatter(buff, space, format, data, formatterData);
    } else {
        // format using printf
        // format must be const char*!!!
        // data must be pointer to the va_list!!!
        written = vsnprintf(buff, space,
                            reinterpret_cast<const char*>(format),
                            *reinterpret_cast<va_list*>(data));
    }
    // check for error
    if (written < 0) return -1;

    // adjust
    if (written >= space) {
        // the message was too long, any non-fitting portion
        // will later be replaced with "..." in makeLogWhere
        return space;
    }
    return written;
}

int makeLogWhere(char *buff, ssize_t space, int written,
                 const char *source, const char *func, size_t line){

    char ibuff[space];

    int iwritten;

    // write position
    iwritten = snprintf(ibuff, space, " {%s:%s():%zd}\n", source, func, line);
    // check for error
    if ((iwritten < 0) || (iwritten >= space)) {
        memcpy(ibuff + space - 4, "...\n", 4);
        iwritten = space;
        //        return total + space;
    }

    if (iwritten > (space - written)) {
        if (space >= iwritten + 3)
            strcpy (buff + space - iwritten - 3, "...");
        memcpy (buff + space - iwritten, ibuff, iwritten);
        return space;
    }
    else {
        memcpy (buff + written, ibuff, iwritten);
        return written + iwritten;
    }

}

int DbgLog_t::logf(unsigned level, FILE *file, const char *source,
                   const char *func, size_t line, const void *format, ...)
{
    va_list ap;
    va_start(ap, format);
    int ret = log(level, file, source, func, line, format, &ap);
    va_end(ap);
    return ret;
}

int DbgLog_t::log(unsigned level, FILE *file, const char *source,
                  const char *func, size_t line, const void *format,
                  void *data, dbglogFormatter_t formatter,
                  void *formatterData)
{
    // ensure we have open file
    if (reopen) {
        ThreadLocker_t tlock(&locker);
        if (logFile >= 0) {
            FileLocker_t flock(logFile);
            openFile();
        } else {
            openFile();
        }
    }

    // check level against mask and get file
    unsigned m = mask;
    if (maskSource) m = *maskSource;
    if ((level & m) && (level != DBG_INTERNAL)) return 0;
    if (!file && (logFile < 0) && !err) return 0;

    char buff[bufSize + 1];
    char msgBuf[bufSize + 1];

    // log message must be formatted outside the thread lock,
    // because formatting might go though Python VM, which
    // could pass the GIL to another thread - if that thread
    // wanted to log something, it would freeze waiting for
    // the thread lock, while the first thread would freeze
    // waiting for the GIL

    // format log message
    int msgLen = makeLogMessage(msgBuf, sizeof(msgBuf), format, data,
                                formatter, formatterData);
    if (msgLen < 0)
        return -1;

    ThreadLocker_t tlock(&locker);

    // format log
    int total = makeLog(buff, sizeof(buff), msgBuf, msgLen,
                        getName(), level, getTimePrecision());
    if (total < 0)
        return -1;

    total = makeLogWhere(buff, sizeof(buff), total, source, func, line);

    if (total < 0)
        return -1;

    // write to log
    if (file) {
        // file given => use stdio to write
        // lock file
        FileLocker_t flock(file);
        // write buffer
        fwrite(buff, 1, total, file);
        // flush file
        fflush(file);
    } else if (logFile >= 0) {
        // write directly to our log file
        // lock file
        FileLocker_t flock(logFile);
        // get bytes to write
        size_t size = total;
        char *start = buff;
        // while we have anything to write
        while (size) {
            // write buffer
            ssize_t written = write(logFile, start, size);
            // break on error
            if (written < 0) break;
            // move to the end of written data
            size -= written;
            start += written;
        }
    }

    // log to stderr
    if (err && !ownsStderr) {
        // lock file
        FileLocker_t flock(stderr);
        // write buffer
        fwrite(buff, 1, total, stderr);
        // flush stderr
        fflush(stderr);
    }

    // OK
    return 0;
}

void DbgLog_t::getMaskValue(char *logMask, size_t len) {
    unsigned m = mask;
    if (maskSource) m = *maskSource;

    switch (m) {
    case MASK_ALL:
        strncpy(logMask, "ALL", len);
        logMask[len] = '\0';
        return;
    case MASK_NONE:
        strncpy(logMask, "NONE", len);
        logMask[len] = '\0';
        return;
    }

    m = ~m;
    char value[32];
    memset(value, 0, 32);
    char *curr = value;

    switch (m & INFO1) {
    case INFO1:
        *curr++ = 'I';
        *curr++ = '1';
        break;
    case INFO2:
        *curr++ = 'I';
        *curr++ = '2';
        break;
    case INFO3:
        *curr++ = 'I';
        *curr++ = '3';
        break;
    case INFO4:
        *curr++ = 'I';
        *curr++ = '4';
        break;
    }

    switch (m & WARN1) {
    case WARN1:
        *curr++ = 'W';
        *curr++ = '1';
        break;
    case WARN2:
        *curr++ = 'W';
        *curr++ = '2';
        break;
    case WARN3:
        *curr++ = 'W';
        *curr++ = '3';
        break;
    case WARN4:
        *curr++ = 'W';
        *curr++ = '4';
        break;
    }

    switch (m & ERR1) {
    case ERR1:
        *curr++ = 'E';
        *curr++ = '1';
        break;
    case ERR2:
        *curr++ = 'E';
        *curr++ = '2';
        break;
    case ERR3:
        *curr++ = 'E';
        *curr++ = '3';
        break;
    case ERR4:
        *curr++ = 'E';
        *curr++ = '4';
        break;
    }

    switch (m & FATAL1) {
    case FATAL1:
        *curr++ = 'F';
        *curr++ = '1';
        break;
    case FATAL2:
        *curr++ = 'F';
        *curr++ = '2';
        break;
    case FATAL3:
        *curr++ = 'F';
        *curr++ = '3';
        break;
    case FATAL4:
        *curr++ = 'F';
        *curr++ = '4';
        break;
    }

    switch (m & DBG1) {
    case DBG1:
        *curr++ = 'D';
        *curr++ = '1';
        break;
    case DBG2:
        *curr++ = 'D';
        *curr++ = '2';
        break;
    case DBG3:
        *curr++ = 'D';
        *curr++ = '3';
        break;
    case DBG4:
        *curr++ = 'D';
        *curr++ = '4';
        break;
    }

    if (curr == value) {
        strncpy(logMask, "NONE", len);
        logMask[len] = '\0';
        return;
    }

    strncpy(logMask, value, len);
}

int DbgLog_t::setOwner(const char *userName, const char *groupName) {
    if (logFile < 0) return -1;

    struct passwd *userInfo = (userName ? getpwnam(userName) : 0);
    struct group *groupInfo = (groupName ? getgrnam(groupName) : 0);

    return fchown(logFile, userInfo ? userInfo->pw_uid : uid_t(-1),
                  groupInfo ? groupInfo->gr_gid : gid_t(-1));
}

int DbgLog_t::setOwner(uid_t user, gid_t group) {
    if (logFile < 0) return -1;

    return fchown(logFile, user, group);
}

// disabled for now
#if 0
extern "C" {
    int pthread_mutex_init __P((pthread_mutex_t *mutex,
                                const pthread_mutexattr_t *mutexattr))
    {
        return 0;
    }

    int pthread_mutex_lock __P((pthread_mutex_t *mutex)) {
        return 0;
    }

    int pthread_mutex_unlock __P((pthread_mutex_t *mutex)) {
        return 0;
    }

    pthread_t pthread_self __P((void)) {
        return pthread_t(0);
    }
}
#endif
