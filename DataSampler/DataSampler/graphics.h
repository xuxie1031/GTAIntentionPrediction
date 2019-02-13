#pragma once
#include "inc\natives.h"
#include "inc\types.h"
#include "inc\enums.h"
#include "inc\main.h"

#include "myTypes.h"
#include "myEnums.h"

#include <string>

void draw_rect(float A_0, float A_1, float A_2, float A_3, int A_4, int A_5, int A_6, int A_7);

void draw_menu_line(std::string caption, float lineWidth, float lineHeight, float lineTop, float lineLeft, float textLeft, bool active, bool title, bool rescaleText = true);

void printOnScreen(std::string s);

void draw_mark_at(Vector2 coords, int r, int g, int b);