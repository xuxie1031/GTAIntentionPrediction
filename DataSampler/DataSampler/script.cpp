#include "script.h"
#include "keyboard.h"
#include "menu.h"

#include <string>
#include <ctime>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <utility>
#include <tuple>
#include <unordered_map>

#include "nlohmann/json.hpp"
using json = nlohmann::json;

#pragma warning(disable : 4244 4305) // double <-> float conversions

Vector3 centerCoords(-1382.03, -390.19, 36.69);
json curSettings;
std::vector<Vehicle> createdVehicles;
std::vector<Ped> createdPeds;
std::vector<Ped> createdDrivers;
std::vector<int> createdSequences;


void printOnScreen(std::string s) {
	DWORD maxTickCount = GetTickCount() + 5000;
	while (GetTickCount() < maxTickCount) {
		draw_menu_line(s, 350.0, 15.0, 15, 15, 5.0, false, true);
		WAIT(0);
	}
	WAIT(0);
}

Entity spawnAtCoords(LPCSTR modelName, int modelType, Vector3 coords, float heading = 0.0)
{
	DWORD model;
	if (modelName[0] != '\0') {
		model = GAMEPLAY::GET_HASH_KEY((char *)modelName);
		STREAMING::REQUEST_MODEL(model);
		while (!STREAMING::HAS_MODEL_LOADED(model)) WAIT(0);
	}
	Entity e;

	if (modelType == MODEL_VEH) {
		Vehicle veh;
		veh = VEHICLE::CREATE_VEHICLE(model, coords.x, coords.y, coords.z, heading, 1, 1);
		VEHICLE::SET_VEHICLE_ON_GROUND_PROPERLY(veh);
		createdVehicles.push_back(veh);
		e = veh;
	}
	else if (modelType == MODEL_PED) {
		Ped p;
		if (modelName[0] != '\0') {
			p = PED::CREATE_PED(PED_TYPE_MISSION, model, coords.x, coords.y, coords.z, heading, FALSE, FALSE);
		}
		else {
			p = PED::CREATE_RANDOM_PED(coords.x, coords.y, coords.z);
		}
		createdPeds.push_back(p);
		e = p;
	}

	if (modelName[0] != '\0') {
		STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(model);
	}

	return e;
}

Ped spawnDriver(Vehicle vehicle, LPCSTR modelName) {
	Ped p;
	DWORD model;

	if (modelName[0] != '\0') {
		model = GAMEPLAY::GET_HASH_KEY((char *)modelName);
		STREAMING::REQUEST_MODEL(model);
		while (!STREAMING::HAS_MODEL_LOADED(model)) WAIT(0);
		p = PED::CREATE_PED_INSIDE_VEHICLE(vehicle, 4, model, -1, 1, 1);
	}
	else {
		p = PED::CREATE_RANDOM_PED_AS_DRIVER(vehicle, true);
	}
	createdDrivers.push_back(p);

	if (modelName[0] != '\0') {
		STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(model);
	}

	return p;
}

void deleteAllCreated() {
	for (int i = createdSequences.size(); i > 0; i--) {
		int sequence = createdSequences.back();
		AI::CLEAR_SEQUENCE_TASK(&sequence);
		createdSequences.pop_back();
	}
	for (int i = createdPeds.size(); i > 0; i--) {
		Entity e = createdPeds[i - 1];
		ENTITY::DELETE_ENTITY(&e);
		createdPeds.pop_back();
	}
	for (int i = createdDrivers.size(); i > 0; i--) {
		Entity e = createdDrivers[i - 1];
		ENTITY::DELETE_ENTITY(&e);
		createdDrivers.pop_back();
	}
	for (int i = createdVehicles.size(); i > 0; i--) {
		Entity e = createdVehicles[i - 1];
		ENTITY::DELETE_ENTITY(&e);
		createdVehicles.pop_back();
	}
}

void deleteCar(int i) {
	int numCars = createdVehicles.size();
	AI::CLEAR_SEQUENCE_TASK(&createdSequences[i]);
	ENTITY::DELETE_ENTITY(&createdDrivers[i]);
	ENTITY::DELETE_ENTITY(&createdVehicles[i]);
	std::swap(createdDrivers[i], createdDrivers[numCars - 1]);
	std::swap(createdVehicles[i], createdVehicles[numCars - 1]);
	std::swap(createdSequences[i], createdSequences[numCars - 1]);

	createdSequences.pop_back();
	createdDrivers.pop_back();
	createdVehicles.pop_back();
}


float randomNum(float low, float high) {
	return low + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (high - low)));
}

int randomNum(int low, int high) {
	return low + rand() % (high - low + 1);
}

float getModelLength(LPCSTR modelName) {
	Vector3 min, max;
	Hash model = GAMEPLAY::GET_HASH_KEY((char *)modelName);
	GAMEPLAY::GET_MODEL_DIMENSIONS(model, &min, &max);
	return max.y - min.y;
}

void clearArea() {
	GAMEPLAY::CLEAR_AREA_OF_PEDS(0, 0, 0, 10000, 1);
	GAMEPLAY::CLEAR_AREA_OF_VEHICLES(0, 0, 0, 10000, false, false, false, false, false);
}

std::string showCoords() {
	Player e = PLAYER::PLAYER_PED_ID();
	Vector3 coords3D = ENTITY::GET_ENTITY_COORDS(e, TRUE);
	Vector2 coords2D;

	GRAPHICS::_WORLD3D_TO_SCREEN2D(coords3D.x, coords3D.y, coords3D.z,
		&(coords2D.x), &(coords2D.y));

	int screen_w, screen_h;
	GRAPHICS::GET_SCREEN_RESOLUTION(&screen_w, &screen_h);
	coords2D.x *= screen_w;
	coords2D.y *= screen_h;

	std::stringstream stream;
	stream << std::fixed << std::setprecision(2) << coords3D.x
		<< "," << coords3D.y << "," << coords3D.z << "," << coords2D.x << "," << coords2D.y << "," << ENTITY::GET_ENTITY_HEADING(e);
	std::string s = stream.str();
	draw_menu_line(s, 350.0, 15.0, coords2D.y, coords2D.x, 5.0, false, true);
	return s;
}

void processSampling() {
	std::ofstream record("sampledCoords.csv");
	WAIT(200);
	while (true) {
		clearArea();
		std::string coords = showCoords();
		if (IsKeyJustUp(VK_NUMPAD5)) {
			// Invoke keyboard
			GAMEPLAY::DISPLAY_ONSCREEN_KEYBOARD(true, "", "", "", "", "", "", 140);
			// Wait for the user to edit
			while (GAMEPLAY::UPDATE_ONSCREEN_KEYBOARD() == 0) WAIT(0);
			// Make sure they didn't exit without confirming their change
			if (!GAMEPLAY::GET_ONSCREEN_KEYBOARD_RESULT()) continue;
			record << GAMEPLAY::GET_ONSCREEN_KEYBOARD_RESULT() << ","  << coords << std::endl;
		}
		if (IsKeyJustUp(VK_NUMPAD0)) {
			record.close();
			WAIT(0);
			break;
		}
		WAIT(0);
	}
}

template<class Terminator>
void processRecording(std::vector<std::string> fileName, Terminator terminator, int waitTime) {

	std::unordered_map<Entity, int> IDMap;

	std::string dirName;
	std::ofstream record;

	int screen_w, screen_h;
	GRAPHICS::GET_SCREEN_RESOLUTION(&screen_w, &screen_h);

	for (int i = 0; i < fileName.size() - 1; i++) {
		if (i != 0) {
			dirName += "/";
		}
		dirName += fileName[i];
		CreateDirectory(dirName.c_str(), NULL);
	}
	record.open(dirName + "/" + fileName.back());

	int count = 0;
	int curID = 0;
	while (true) {
		std::vector<bool> outOfScreen;
		for (int i = 0; i < createdVehicles.size(); i++) {
			Entity e = createdVehicles[i];
			if (!IDMap.count(e)) {
				IDMap[e] = curID;
				curID++;
			}

			Vector3 curCoords = ENTITY::GET_ENTITY_COORDS(e, !ENTITY::IS_ENTITY_DEAD(e));
			Vector2 coords2D;
			GRAPHICS::_WORLD3D_TO_SCREEN2D(curCoords.x, curCoords.y, curCoords.z, &(coords2D.x), &(coords2D.y));

			if (coords2D.x < 0) {
				outOfScreen.push_back(true);
				continue;
			}
			outOfScreen.push_back(false);

			coords2D.x *= screen_w;
			coords2D.y *= screen_h;

			record << count << "," << IDMap[e] << "," << coords2D.x << "," << coords2D.y << ",Veh" << std::endl;
		}

		for (int i = 0; i < createdPeds.size(); i++) {
			Entity e = createdPeds[i];
			if (!IDMap.count(e)) {
				IDMap[e] = curID;
				curID++;
			}
			Vector3 curCoords = ENTITY::GET_ENTITY_COORDS(e, !ENTITY::IS_ENTITY_DEAD(e));
			Vector2 coords2D;
			GRAPHICS::_WORLD3D_TO_SCREEN2D(curCoords.x, curCoords.y, curCoords.z, &(coords2D.x), &(coords2D.y));

			if (coords2D.x < 0) {
				continue;
			}

			coords2D.x *= screen_w;
			coords2D.y *= screen_h;

			record << count << "," << IDMap[e] << "," << coords2D.x << "," << coords2D.y << ",Ped" << std::endl;

		}

		if (terminator()) { 
			break;
		}
		count++;
		WAIT(waitTime);
	}
}

void teleport() {
	Ped p = PLAYER::PLAYER_PED_ID();
	ENTITY::SET_ENTITY_COORDS(p, centerCoords.x, centerCoords.y, centerCoords.z, 1, 0, 0, 1);
}

void complexScene() {
	int waitTime = 100;

	int totalCars = 3;
	std::vector<std::pair<Vector3, Vector3>> startPositions{ std::make_pair(Vector3(-1420.16,-421.93,36.22), Vector3(-1427.47,-426.17,36.05)),
															 std::make_pair(Vector3(-1417.19,-426.05,36.14), Vector3(-1425.10,-430.49,36.00))
	};
	std::vector<std::vector<Vector3>> validGoals{ { Vector3(-1429.17,-327.34,44.43), Vector3(-1320.46,-361.92,36.75) },
												  { Vector3(-1320.16,-489.37,33.34), Vector3(-1319.56, -367.42, 36.69)} };
	float heading = 302.37;
	float carInterval = 2.0;
	float stopRange = 2.0;
	int driveStyle = 786475; // normal except 128--stop on lights
	float minSpeed = 3.0;
	float maxSpeed = 10.0;

	int numWanderPeds = 6;
	std::tuple<Vector3, Vector3, Vector3> genArea = std::make_tuple(Vector3(-1362.92, -394.49, 36.55), Vector3(-1387.76, -411.07, 36.56), Vector3(-1378.92, -370.83, 36.51));

	int numScriptPeds = 3;
	std::vector<std::vector<Vector3>> endPoints{ { Vector3(-1377.41,-366.81,36.60), Vector3(-1380.84,-366.67,36.57) } , { Vector3(-1409.80,-386.40,36.63), Vector3(-1408.43,-385.78,36.61) }, { Vector3(-1390.06,-415.37,36.58), Vector3(-1385.91,-417.79,36.61) }, { Vector3(-1359.15,-397.74,36.61), Vector3(-1357.56,-393.77,36.58) } };

	char carModel[] = "Adder";
	char pedModel[] = "";

	auto setCars = [&]() {

		float carLength = getModelLength(carModel);

		std::vector<int> numCars(startPositions.size(), 0);
		for (int i = 0; i < totalCars; i++) {
			int lane = randomNum(0, numCars.size() - 1);
			numCars[lane]++;
		}

		for (int i = 0; i < startPositions.size(); i++) {
			Vector3 start;
			Vector3 front = startPositions[i].first;
			Vector3 back = startPositions[i].second;
			start.x = randomNum(front.x, back.x);
			float ratio = (start.x - front.x) / (back.x - front.x);
			start.y = (back.y - front.y) * ratio + front.y;
			start.z = (back.z - front.z) * ratio + front.z;

			ratio = (carInterval + carLength) / SYSTEM::VDIST(front.x, front.y, front.z, back.x, back.y, back.z);

			for (int j = 0; j < numCars[i]; j++) {

				Vehicle v = spawnAtCoords(carModel, MODEL_VEH, start, heading);
				Ped driver = spawnDriver(v, pedModel);

				int goal = randomNum(0, validGoals[i].size() - 1);
				float speed = randomNum(minSpeed, maxSpeed);

				int sequence = 0;
				AI::OPEN_SEQUENCE_TASK(&sequence);
				AI::TASK_VEHICLE_DRIVE_TO_COORD(0, v, validGoals[i][goal].x, validGoals[i][goal].y, validGoals[i][goal].z, speed, 0, GAMEPLAY::GET_HASH_KEY(carModel), driveStyle, stopRange, 1.0);
				AI::CLOSE_SEQUENCE_TASK(sequence);
				AI::TASK_PERFORM_SEQUENCE(driver, sequence);

				createdSequences.push_back(sequence);

				start.x = (back.x - front.x) * ratio + start.x;
				start.y = (back.y - front.y) * ratio + start.y;
				start.z = (back.z - front.z) * ratio + start.z;
				WAIT(0);
			}
		}
	};

	auto setWanderPed = [&]() {

		std::vector<Vector3> pedsPosition;

		Vector3 origin = std::get<0>(genArea);
		Vector3 unit1 = std::get<1>(genArea);
		Vector3 unit2 = std::get<2>(genArea);

		float walkRadius = SYSTEM::VDIST(unit1.x, unit1.y, unit1.z, unit2.x, unit2.y, unit2.z) / 2;

		for (int i = 0; i < numWanderPeds; i++) {
			float ratio1 = randomNum(0.0f, 1.0f);
			float ratio2 = randomNum(0.0f, 1.0f);
			float x = ratio1 * (unit1.x - origin.x) + ratio2 * (unit2.x - origin.x) + origin.x;
			float y = ratio1 * (unit1.y - origin.y) + ratio2 * (unit2.y - origin.y) + origin.y;
			float z = ratio1 * (unit1.z - origin.z) + ratio2 * (unit2.z - origin.z) + origin.z;
			bool tooClose = false;
			for (int j = 0; j < pedsPosition.size(); j++) {
				if (SYSTEM::VDIST(pedsPosition[j].x, pedsPosition[j].y, pedsPosition[j].z, x, y, z) < 1.0) {
					tooClose = true;
					break;
				}
			}
			if (tooClose) {
				i--;
				continue;
			}
			pedsPosition.push_back(Vector3(x, y, z));
			Ped p = spawnAtCoords(pedModel, MODEL_PED, pedsPosition.back());
			AI::TASK_WANDER_IN_AREA(p, centerCoords.x, centerCoords.y, centerCoords.z, walkRadius, 0.0, 0.0);
			//AI::TASK_WANDER_STANDARD(p, 10.0, 10);
			WAIT(0);
		}
	};
	
	auto setScriptPed =  [&]() {
		std::vector<std::pair<Vector3, Vector3>> pedRoutes;

		for (int i = 0; i < numScriptPeds; i++) {
			int startArea = randomNum(0, endPoints.size() - 1);
			int startCode = randomNum(0, endPoints[startArea].size() - 1);
			Vector3 start = endPoints[startArea][startCode];

			int endArea = randomNum(0, endPoints.size() - 1);
			if (endArea == startArea) {
				i--;
				continue;
			}
			int endCode = randomNum(0, endPoints[endArea].size() - 1);
			Vector3 end = endPoints[endArea][endCode];

			bool repeated = false;
			for (int j = 0; j < pedRoutes.size(); j++) {
				if (start == pedRoutes[j].first || (end == pedRoutes[j].first && start == pedRoutes[j].second)) {
					repeated = true;
					break;
				}
			}

			if (repeated) {
				i--;
				continue;
			}

			pedRoutes.push_back(std::make_pair(start, end));
			Ped p = spawnAtCoords(pedModel, MODEL_PED, start);
			AI::TASK_GO_STRAIGHT_TO_COORD(p, end.x, end.y, end.z, 1.0, 200000, 0.0, 0.2);
			WAIT(0);
		}

	};

	auto terminator = []() {
		clearArea();
		for (int i = 0; i < createdVehicles.size(); i++) {
			if (AI::GET_SEQUENCE_PROGRESS(createdDrivers[i]) == -1) {
				deleteCar(i);
				i--;
			}
		}
		if (createdVehicles.size() == 0) {
			return true;
		}
		return false;
	};

	clearArea();
	setCars();
	setWanderPed();
	setScriptPed();
	processRecording({ "complexScene", "record1.csv" }, terminator, waitTime);
	deleteAllCreated();
}

void toggleCamera() {
	int cameraHeight = 15.0;
	Vector3 cameraRot(270.0, 360.0 - 302.37, 0.0);

	static bool initialized(false);
	static bool active(false);

	static Camera customCam = CAM::CREATE_CAM("DEFAULT_SCRIPTED_CAMERA", true);

	if (!initialized) {
		initialized = true;
		CAM::SET_CAM_COORD(customCam, centerCoords.x, centerCoords.y, centerCoords.z + cameraHeight);
		CAM::SET_CAM_FOV(customCam, 130.0);
		// CAM::POINT_CAM_AT_COORD(customCam, centerCoords.x, centerCoords.y, centerCoords.z);
		CAM::SET_CAM_ROT(customCam, cameraRot.x, cameraRot.y, cameraRot.z, 2);
		CAM::SET_CINEMATIC_MODE_ACTIVE(false);
	}
	if (!active) {
		CAM::SET_CAM_ACTIVE(customCam, true);
		CAM::RENDER_SCRIPT_CAMS(true, false, 3000, true, false);
	}
	else {
		CAM::SET_CAM_ACTIVE(customCam, false);
		CAM::RENDER_SCRIPT_CAMS(false, false, 3000, true, false);
	}
	active = !active;
}

// TODO: enable multiple setting reading
// Need new menu
void loadSettings() {
	std::ifstream config("settings.json");
	curSettings << config;
}

// TODO: Guide through creation
void createSettings() {

	json settings;
	settings["cameraHeight"] = 15.0;
	settings["centerCoords"] = { -1382.03, -390.19, 36.69 };
	settings["period"] = 100;
	settings["totalCars"] = 3;
	settings["carStartPositions"] = { { { "front", {-1420.16,-421.93,36.22} }, { "back", {-1427.47,-426.17,36.05} } }, { { "front", {-1417.19,-426.05,36.14} }, { "back", {-1425.10,-430.49,36.00} } } };
	settings["carHeading"] = 302.37;
	settings["carInterval"] = 2.0;
	settings["carStopRange"] = 2.0;
	settings["driveStyle"] = 786475; // normal except 128--stop on lights
	settings["carMinSpeed"] = 3.0;
	settings["carMaxSpeed"] = 10.0;
	settings["totalWanderPeds"] = 6;
	settings["WanderPedsGenArea"] = { { "origin", {-1362.92, -394.49, 36.55} }, { "unit1", {-1387.76, -411.07, 36.56} }, { "unit2", {-1378.92, -370.83, 36.51} } };
	settings["totalScriptPeds"] = 3;
	settings["ScriptPedsEndPoints"] = { { {-1377.41,-366.81,36.60}, {-1380.84,-366.67,36.57} }, { {-1409.80,-386.40,36.63}, {-1408.43,-385.78,36.61} }, { {-1390.06,-415.37,36.58}, {-1385.91,-417.79,36.61} }, { {-1359.15,-397.74,36.61}, {-1357.56,-393.77,36.58} } };
	settings["carModel"] = "Adder";
	settings["pedModel"] = "";

	std::ofstream config("settings.json");
	config << settings;
	config.close();
}

Menu mainMenu("Main Menu", std::vector<std::string>
{"Sample Points",
"Teleport to Scene",
"Toggle Camera",
"Complex Scene",
"Create Settings"});

void processMainMenu() {
	int lineActive = 0;

	DWORD waitTime = 150;

	while (true) {
		DWORD maxTickCount = GetTickCount() + waitTime;
		do
		{
			mainMenu.drawVertical(lineActive);
			WAIT(0);
		} while (GetTickCount() < maxTickCount);

		waitTime = 0;

		bool bSelect, bBack, bUp, bDown;
		get_button_state(&bSelect, &bBack, &bUp, &bDown, NULL, NULL);
		if (bSelect)
		{
			switch (lineActive)
			{
			case 0:
				processSampling();
				break;
			case 1:
				teleport();
				break;
			case 2:
				toggleCamera();
				break;
			case 3:
				complexScene();
				break;
			case 4:
				createSettings();
			}
			waitTime = 200;
		}
		else if (bBack) {
			break;
		}
		else if (bUp) {
			if (lineActive == 0)
				lineActive = mainMenu.lineCount();
			lineActive--;
			waitTime = 150;
		}
		else if (bDown)
		{
			lineActive++;
			if (lineActive == mainMenu.lineCount())
				lineActive = 0;
			waitTime = 150;
		}
	}
}


void main() {
	while (true) {
		if (IsKeyJustUp(VK_F6)) {
			AUDIO::PLAY_SOUND_FRONTEND(-1, "NAV_UP_DOWN", "HUD_FRONTEND_DEFAULT_SOUNDSET", 0);
			processMainMenu();
		}
		WAIT(0);
	}
}

void ScriptMain() {
	srand(GetTickCount());
	main();
}
