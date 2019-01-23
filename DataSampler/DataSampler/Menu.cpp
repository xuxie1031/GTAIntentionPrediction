#include "Menu.h"

int NUMBER_OF_LINES_SHOW = 15;

std::string line_as_str(std::string text, bool *pState)
{
	while (text.size() < 18) text += " ";
	return text + (pState ? (*pState ? " [ON]" : " [OFF]") : "");
}

Menu::Menu(const std::string& caption, const std::vector<std::string>& lines, const std::vector<bool*>& states) {
	mCaption = caption;
	mLines = lines;
	mMaxWidth = 250;
	for (int i = 0; i < mLines.size(); i++) {
		size_t lineLen = mLines[i].length() * 15;
		if (lineLen > mMaxWidth) {
			mMaxWidth = lineLen;
		}
	}

	if (states.size() > 0) {
		mMaxWidth += 100;
		mStates = states;
	}
	else {
		mStates = std::vector<bool*>(mLines.size(), nullptr);
	}
}

size_t Menu::lineCount() {
	return mLines.size();
}

void Menu::drawVertical(int lineActive) {
	int end = mLines.size();
	int start = 0;
	lineActive++;
	if (end > NUMBER_OF_LINES_SHOW) {
		if (lineActive > NUMBER_OF_LINES_SHOW / 2) {
			start = min(end - NUMBER_OF_LINES_SHOW, lineActive - NUMBER_OF_LINES_SHOW / 2);
		}
		end = start + NUMBER_OF_LINES_SHOW;
	}
	lineActive--;
	draw_menu_line(mCaption, mMaxWidth, 15.0, 18.0, 0.0, 5.0, false, true);
	for (int i = start; i < end; i++)
		if (i != lineActive)
			draw_menu_line(line_as_str(mLines[i], mStates[i]),
				mMaxWidth, 9.0, 60.0 + (i - start) * 36.0, 0.0, 9.0, false, false);

	if (mLines.size() > 0) {
		draw_menu_line(line_as_str(mLines[lineActive], mStates[lineActive]),
			mMaxWidth + 1.0, 11.0, 56.0 + (lineActive - start) * 36.0, 0.0, 7.0, true, false);
	}
}