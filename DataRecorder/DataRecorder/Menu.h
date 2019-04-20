#pragma once

#include <string>
#include <vector>
#include <functional>
#include <cmath>

#include "inc\natives.h"
#include "inc\types.h"
#include "inc\enums.h"
#include "inc\main.h"

#include "myEnums.h"
#include "myTypes.h"

#include "graphics.h"
#include "keyboard.h"

class Menu {
public:
	Menu(const std::string& CAPTION, const std::vector<MenuItem>& ITEMS, bool SINGLE_USE = false);
	size_t lineCount();
	void drawVertical(int lineActive);
	void addMenuItem(const MenuItem& newItem);
	void processMenu();
	bool Menu::onTick();
	void Menu::deleteItem(int i);

	int lineActive;
	bool singleUse;
	std::string caption;
	std::vector<MenuItem> items; // TODO: hide this

private:
	static const int NUMBER_OF_LINES_SHOW;
	static const float ITEM_CHAR_WIDTH;
	static const float CAPTION_CHAR_WIDTH;
	static const float DESCRIPTION_EXTRA_WIDTH;
	float maxWidth;

	std::string makeLine(std::string text, bool *pState);
	bool isItemActive(int i);
	bool isMenuActive();
	float lineWidth(const std::string& line, bool isCaption, bool isDescription);

};