#pragma once

#include <vector>

#include "inc\natives.h"
#include "inc\types.h"
#include "inc\enums.h"
#include "inc\main.h"


class InstructionButtons {
public:
	static InstructionButtons* getInstance();
	void loadButtonList(const std::vector<std::pair<eControl, std::string>>& buttons);
	void render();

private:
	InstructionButtons();
	int handle;
	static InstructionButtons* instance;
};
