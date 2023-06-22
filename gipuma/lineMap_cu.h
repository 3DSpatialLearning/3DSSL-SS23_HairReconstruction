#pragma once

#include "config.h"
#include "line_cu.h"
#include "managed.h"

class __align__(128) LineMap_cu : public Managed {
   public:
    Line_cu lines[MAX_IMAGES];
    LineMap_cu() {}
    ~LineMap_cu() {}
};
