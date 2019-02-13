/*
	THIS FILE IS A PART OF GTA V SCRIPT HOOK SDK
				http://dev-c.com
			(C) Alexander Blade 2015
			*/
#pragma once

#include "inc\natives.h"
#include "inc\types.h"
#include "inc\enums.h"

#include "inc\main.h"

#include "myEnums.h"
#include "myTypes.h"
#include "graphics.h"
#include "keyboard.h"
#include "Menu.h"
#include "GameResources.h"
#include "debug.h"
// #include "Settings.h"

#include <string>
#include <ctime>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <utility>
#include <tuple>
#include <unordered_map>
#include <Windows.h>
#include <functional>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

void ScriptMain();
