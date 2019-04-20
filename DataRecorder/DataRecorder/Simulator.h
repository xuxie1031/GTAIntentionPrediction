#pragma once

#include "keyboard.h"
#include "graphics.h"
#include "SimulationData.h"
#include "gamePlay.h"
#include "gameResources.h"

class Simulator {
public:
	Simulator(SimulationData& data);
	void startSimulation();
	void processRecording(std::function<bool()> delegate, std::string fileName);
	void processReplay(bool drawRainbow);
private:
	void setCars(SimulationData::VehSettings& settings);
	void setPeds(SimulationData::PedSettings& settings);
	void loadPredictions(std::unordered_map<int, std::unordered_map<int, std::vector<std::vector<Vector2>>>>& coordsMap);
	void loadReplay(std::unordered_map<int, std::unordered_map <int, std::pair<Vector3d, float>>>& coordsMap, std::unordered_map<int, int>& lastAppear);
	SimulationData& data;
};