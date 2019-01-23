#pragma once
#include <string>
#include <vector>
#include "script.h"
#include "graphics.h"

class Menu {
public:
	Menu(const std::string&, const std::vector<std::string>&, const std::vector<bool*>& = std::vector<bool*>());
	size_t lineCount();
	void drawVertical(int);
private:
	int mMaxWidth;
	std::string mCaption;
	std::vector<std::string> mLines;
	std::vector<bool*> mStates;
};