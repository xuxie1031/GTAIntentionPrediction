#include "debug.h"

void outputDebugMessage(std::string s) {
	static bool initialized = false;
	if (initialized = false) {
		initialized = true;
		remove("debug.txt");
	}

	std::ofstream debug("debug.txt", std::fstream::app);
	DWORD t = GetTickCount();

	debug << t << ": " << s << std::endl;
	debug.close();
}