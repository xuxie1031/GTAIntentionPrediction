#include "debug.h"

void outputDebugMessage(std::string s) {

	std::string logFile = "DataSampler.log";
	std::ofstream debug;

	static bool initialized = false;
	if (initialized = false) {
		initialized = true;
		debug.open(logFile);
	}
	else {
		debug.open(logFile, std::fstream::app);
	}
	DWORD t = GetTickCount();

	debug << t << ": " << s << std::endl;
	debug.close();
}