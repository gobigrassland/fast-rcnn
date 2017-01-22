/*
 * $Id: dbglogobject.h 25 2009-03-26 16:15:16Z burlog $
 *
 * DBGLOG -- debug and runtime logger
 *
 * AUTHOR: Vaclav Blazek <blazek@firma.seznam.cz>
 *
 */

#ifndef DBGLOG_DBG_H_
#define DBGLOG_DBG_H_

#ifndef __GNUC__
#error "Oops! You must use GCC to compile this!"
#endif

#include "dbglog.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include <unistd.h>
#include <sys/file.h>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/stat.h>

#include <pthread.h>
#include <errno.h>


// disabled for now
#if 0
extern "C" {
    int pthread_mutex_init __P((pthread_mutex_t *MUTEX,
                                const pthread_mutexattr_t *MUTEXATTR))
        __attribute__ ((weak));

    int pthread_mutex_lock __P((pthread_mutex_t *mutex))
        __attribute__ ((weak));

    int pthread_mutex_unlock __P((pthread_mutex_t *mutex))
        __attribute__ ((weak));

    pthread_t pthread_self __P((void))
        __attribute__ ((weak));
}
#endif

const unsigned int MAX_APP_NAME = 256;

class TID_t {
public:
    TID_t(pthread_t id = 0, const char *name = 0, TID_t *next = 0)
        : next(next), id(id)
    {
        setName(name);
    }

    void setName(const char *newName) {
        if (newName == PID_APPNAME) {
            nameIsPid = true;
            nameIsPidTid = false;
        } else if (newName == PID_TID_APPNAME) {
            nameIsPid = false;
            nameIsPidTid = true;
        } else {
            nameIsPid = false;
            nameIsPidTid = false;
            if (newName) {
                strncpy(name, newName, MAX_APP_NAME);
                name[MAX_APP_NAME] = '\0';
            } else name[0] = '\0';
        }
    }

    const char* getName() const {
        if (nameIsPid) return PID_APPNAME;
        if (nameIsPidTid) return PID_TID_APPNAME;
        return name;
    }

    ~TID_t() {
        delete next;
    }

    TID_t *next;

    pthread_t id;
    char name[MAX_APP_NAME + 1];
    bool nameIsPid;
    bool nameIsPidTid;
};

class DbgLog_t {
public:
    DbgLog_t()
        : logFile(-1), curFileName(NULL),
          mask(~(INFO3 | WARN2 | ERR2 | FATAL1)), err(false),
          locker(initializer), ids(NULL), maskSource(0),
          bufSize(DEFAULT_BUF_SIZE), ownsStderr(false),
          precision(0), precisionSource(0), reopen(false)
    {
        // initialize mutex
        pthread_mutex_init(&locker, 0);
    }

    ~DbgLog_t() {
        if (logFile >= 0) close(logFile);
        if (ownsStderr) close(STDERR_FILENO);
        if (curFileName) free(curFileName);
        delete ids;
    }

    /** Explicit initialization of log object. */
    void init() {
        /* get values */
        const char *logFile = getenv("LOG_FILE");
        if (!logFile) logFile = getenv("MDBG_FILE");
        const char *logMask = getenv("LOG_MASK");
        if (!logMask) logMask = getenv("MDBG_MASK");
        const char *logStderr = getenv("LOG_STDERR");
        if (!logStderr) logStderr = getenv("MDBG_STDERR");

        const char *logOwner = getenv("LOG_OWNER");

        // open file
        setFile(logFile);
        // set log mask
        setMask(logMask);

        // stderr is off by default
        useStderr(false);
        if (logStderr) {
            if (*logStderr && strcmp("0", logStderr))
                useStderr(true);
        }

        if (logOwner) {
            const char* colon = strchr(logOwner, ':');
            char *user = 0;
            char *group = 0;
            if (colon) {
                if (colon == logOwner) {
                    if (*(colon + 1))
                        group = strdup(colon + 1);
                } else {
#ifdef __GLIBC__
                    user = strndup(logOwner, colon - logOwner);
#else
                    user = static_cast<char *>(malloc(colon - logOwner));
                    strncpy(user, logOwner, colon - logOwner);
#endif
                    if (*(colon + 1))
                        group = strdup(colon + 1);
                }
            } else {
                user = strdup(logOwner);
            }

            setOwner(user, group);
            if (user) free(user);
            if (group) free(group);
        }

        const char *sbufSize = getenv("LOG_BUFSIZE");
        if (sbufSize) {
            int nbufSize = atoi(sbufSize);
            if (nbufSize > 0) bufSize = nbufSize;
        }
    }

    int setFile(const char *fileName);

    int reopenFile() {
        reopen = true;
        return 0;
    }

    int openFile();

    inline const char* getLogFile() const {
        return curFileName;
    }

    void useStderr(bool err) {
        ThreadLocker_t tlock(&locker);
        this->err = err;
    }

    bool getUseStderr() const {
        return err;
    }

    void bindStderr(bool val) {
        if (val && !ownsStderr) {
            // we are usurping stderr
            if (logFile >= 0)
                dup2(logFile, STDERR_FILENO);
        } else if (!val && ownsStderr) {
            // we are loosing stderr
            int f = open("/dev/null", O_WRONLY | O_APPEND);
            dup2(f, STDERR_FILENO);
            close(f);
        }
        ownsStderr = val;
    }

    bool checkLevel(int level) {
        unsigned m = mask;
        if (maskSource) m = *maskSource;
        return !(level & m);
    }

    void setMask(const char *maskStr);

    inline void setMaskSource(unsigned *maskSource) {
        if (maskSource) {
            if (!this->maskSource) {
                this->maskSource = maskSource;
                *this->maskSource = mask;
            } else {
                *maskSource = mask;
                this->maskSource = maskSource;
            }
        } else {
            if (this->maskSource) {
                mask = *this->maskSource;
                this->maskSource = NULL;
            }
        }
    }

    inline unsigned int getTimePrecision() {
        if (precisionSource) return *precisionSource;
        return precision;
    }

    inline void setTimePrecision(unsigned int precision) {
        if (precisionSource) *precisionSource = precision;
        else this->precision = precision;
    }

    inline void setTimePrecisionSource(unsigned int *precisionSource) {
        if (precisionSource) {
            if (!this->precisionSource) {
                this->precisionSource = precisionSource;
                *this->precisionSource = precision;
            } else {
                *precisionSource = precision;
                this->precisionSource = precisionSource;
            }
        } else {
            if (this->precisionSource) {
                precision = *this->precisionSource;
                this->precisionSource = NULL;
            }
        }
    }

    inline void setAppName(const char *name) {
        ThreadLocker_t tlock(&locker);
        pthread_t self = pthread_self();
        TID_t *id = ids;
        for (; id != NULL; id = id->next)
            if (id->id == self) break;

        if (id != NULL) {
            id->setName(name);
        } else {
            ids = new TID_t(self, name, ids);
        }
    }

    inline const char* getAppName() {
        ThreadLocker_t tlock(&locker);
        pthread_t self = pthread_self();
        TID_t *id = ids;
        for (; id != NULL; id = id->next)
            if (id->id == self) break;

        if (!id) return "";
        return id->getName();
    }

    static DbgLog_t *defLog;

    int logf(unsigned level, FILE *file, const char *source, const char *func,
             size_t line, const void *format, ...);

    int log(unsigned level, FILE *file, const char *source, const char *func,
            size_t line, const void *format, void *data,
            dbglogFormatter_t formatter = 0, void *formatterData = 0);

    void getMaskValue(char *logMask, size_t len);

    int setOwner(const char *user, const char *group);
    int setOwner(uid_t user, gid_t group);

    void setBufSize(unsigned bufSize) {
        this->bufSize = bufSize;
    }

    static const unsigned MASK_NONE  = 0xffffffff;
    static const unsigned MASK_ALL   = ~MASK_NONE;

    static void initializeDefLog();

private:
    DbgLog_t(const DbgLog_t&);
    DbgLog_t operator=(const DbgLog_t&);

    struct ThreadLocker_t {
        ThreadLocker_t(pthread_mutex_t *mutex)
            : mutex(mutex)
        {
            pthread_mutex_lock(mutex);
        }

        ~ThreadLocker_t() {
            pthread_mutex_unlock(mutex);
        }

        pthread_mutex_t *mutex;
    };

    inline const char* getName() {
        pthread_t self = pthread_self();
        TID_t *id = ids;
        for (; id != NULL; id = id->next)
            if (id->id == self)
                return id->getName();
        return 0;
    }

    int logFile;
    char *curFileName;
    unsigned mask;

    bool err;

    pthread_mutex_t locker;
    TID_t *ids;

    unsigned *maskSource;

    static const unsigned DEFAULT_BUF_SIZE = 1024;

    unsigned bufSize;

    static const pthread_mutex_t initializer;

    bool ownsStderr;

    unsigned int precision;

    unsigned int *precisionSource;

    int reopen;
};

#endif // DBGLOG_DBG_H_
