#pragma once

#include "myTypes.h"

#include <string>
#include <vector>

class Settings {
public:
	struct{
		Vector3 spectatePosition; // player will be teleported to this position once simulation starts. If player is too far away, some functionalities fails.
		int numSimulations;
		int recordInterval; // interval between two recordings in ms
		std::vector<std::string> recordDirectory;
	}  recording;

	struct {
		float height;
		float fov;
		Vector3d position; // position relative to ground
		Vector3d rotation;
	} camera;

	// TODO: organize settings in a way that:
	// 1. has randomness
	// 2. general
	struct scriptVehicle {

	};

	struct pedestrian {

	};
	struct replay {
		std::string replayFile;
		std::string predictionFile;
	};

	Settings(); // create custom new settings
	Settings(std::string settingsFile);
	void saveSettingsToFile(std::string fileName = "");
	void loadSettings(std::string settingsFile);
	void clearSettings();

private:

};