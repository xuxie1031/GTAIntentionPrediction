#include "Menu.h"

int NUMBER_OF_LINES_SHOW = 15;

std::string Menu::makeLine(std::string text, bool *pState)
{
	while (text.size() < 18) text += " ";
	return text + (pState ? (*pState ? " [ON]" : " [OFF]") : "");
}

Menu::Menu(const std::string& caption, const std::vector<std::string>& lines, const std::vector<std::function<void()>>& functions, const std::vector<bool*>& states) {
	mCaption = caption;
	mFunctions = functions;
	mLines = lines;
	mMaxWidth = 250;
	size_t lineLen;

	lineLen = caption.length() * 15;
	if (lineLen > mMaxWidth) {
		mMaxWidth = lineLen;
	}

	for (int i = 0; i < mLines.size(); i++) {
		lineLen = mLines[i].length() * 15;
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
			draw_menu_line(makeLine(mLines[i], mStates[i]),
				mMaxWidth, 9.0, 60.0 + (i - start) * 36.0, 0.0, 9.0, false, false);

	if (lineCount() > 0) {
		draw_menu_line(makeLine(mLines[lineActive], mStates[lineActive]),
			mMaxWidth + 1.0, 11.0, 56.0 + (lineActive - start) * 36.0, 0.0, 7.0, true, false);
	}
}

void Menu::processMenu() {
	int lineActive = 0;

	DWORD waitTime = 150;

	while (true) {
		DWORD maxTickCount = GetTickCount() + waitTime;
		do
		{
			drawVertical(lineActive);
			WAIT(0);
		} while (GetTickCount() < maxTickCount);

		waitTime = 0;

		bool bSelect, bBack, bUp, bDown;
		get_button_state(&bSelect, &bBack, &bUp, &bDown, NULL, NULL);
		if (bSelect)
		{
			mFunctions[lineActive]();
			waitTime = 200;
		}
		else if (bBack) {
			break;
		}
		else if (bUp) {
			if (lineActive == 0)
				lineActive = lineCount();
			lineActive--;
			waitTime = 150;
		}
		else if (bDown)
		{
			lineActive++;
			if (lineActive == lineCount())
				lineActive = 0;
			waitTime = 150;
		}
	}
}