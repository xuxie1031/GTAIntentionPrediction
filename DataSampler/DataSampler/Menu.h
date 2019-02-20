#pragma once
#include <string>
#include <vector>
#include <functional>

#include "graphics.h"
#include "keyboard.h"

class Menu {
public:
	Menu(const std::string& caption, const std::vector<std::string>& lines, const std::vector<std::function<void()>>& functions, const std::vector<bool*>& states = {});
	size_t lineCount();
	void drawVertical(int lineActive);
	void processMenu();
	bool oneTimeSelection;

private:
	std::string makeLine(std::string text, bool *pState);

	int mMaxWidth;
	std::string mCaption;
	std::vector<std::string> mLines;
	std::vector<bool*> mStates;
	std::vector<std::function<void()>> mFunctions;
};