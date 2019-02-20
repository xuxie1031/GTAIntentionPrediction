#include "script.h"

#pragma warning(disable : 4244 4305) // double <-> float conversions

const int PED_ID_OFFSET = 100;
const std::string settingsDirectory = "RecordSettings";

json settings;
std::string settingFile;

std::vector<std::string> readAllFiles(std::string folder)
{
	std::vector<std::string> res;
	std::string path = folder + "\\*";
	WIN32_FIND_DATA data;
	HANDLE hFind = FindFirstFile(path.c_str(), &data);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			if (!(data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				res.push_back(folder + "/" + data.cFileName);
			}
		} while (FindNextFile(hFind, &data));
		FindClose(hFind);
	}
	return res;
}

float randomNum(float low, float high) {
	return low + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (high - low)));
}

int randomNum(int low, int high) {
	return low + rand() % (high - low + 1);
}

float getModelLength(LPCSTR modelName) {
	Vector3d min, max;
	Hash model = GAMEPLAY::GET_HASH_KEY((char *)modelName);
	STREAMING::REQUEST_MODEL(model);
	while (!STREAMING::HAS_MODEL_LOADED(model)) WAIT(0);
	GAMEPLAY::GET_MODEL_DIMENSIONS(model, &min, &max);
	return max.y - min.y;
}

void resetPlayer() {
	ENTITY::SET_ENTITY_COLLISION(PLAYER::PLAYER_PED_ID(), 1, 1);
	PED::SET_PED_GRAVITY(PLAYER::PLAYER_PED_ID(), 1);
}

void clearArea() {
	GAMEPLAY::CLEAR_AREA_OF_PEDS(0, 0, 0, 10000, 1);
	GAMEPLAY::CLEAR_AREA_OF_VEHICLES(0, 0, 0, 10000, false, false, false, false, false);
}

void getEntityMotion(Entity e, Vector3d* coords3D = NULL, Vector2* coords2D = NULL, float* heading = NULL, float* speed = NULL) {

	Vector3d coords = ENTITY::GET_ENTITY_COORDS(e, TRUE);

	if (coords3D != NULL) {
		*coords3D = coords;
	}

	if (coords2D != NULL) {
		GRAPHICS::_WORLD3D_TO_SCREEN2D(coords.x, coords.y, coords.z, &(coords2D->x), &(coords2D->y));
		int screen_w, screen_h;
		GRAPHICS::GET_SCREEN_RESOLUTION(&screen_w, &screen_h);
		coords2D->x *= screen_w;
		coords2D->y *= screen_h;
	}

	if (heading != NULL) {
		*heading = ENTITY::GET_ENTITY_HEADING(e);
	}

	if (speed != NULL) {
		*speed = ENTITY::GET_ENTITY_SPEED(e);
	}

}

std::string showPlayerCoords() {
	Player e = PLAYER::PLAYER_PED_ID();
	Vector3d coords3D;
	Vector2 coords2D;
	float heading;

	getEntityMotion(e, &coords3D, &coords2D, &heading);

	std::stringstream stream;
	stream << std::fixed << std::setprecision(2) << coords3D.x
		<< "," << coords3D.y << "," << coords3D.z << "," << coords2D.x << "," << coords2D.y << "," << heading;
	std::string s = stream.str();
	draw_menu_line(s, 350.0, 15.0, coords2D.y, coords2D.x, 5.0, false, true);
	return s;
}

std::string getTextInput() {
	std::string res;
	// Invoke keyboard
	GAMEPLAY::DISPLAY_ONSCREEN_KEYBOARD(true, "", "", "", "", "", "", 140);
	// Wait for the user to edit
	while (GAMEPLAY::UPDATE_ONSCREEN_KEYBOARD() == 0) WAIT(0);
	// Make sure they didn't exit without confirming their change
	if (!GAMEPLAY::GET_ONSCREEN_KEYBOARD_RESULT()) {
		return res;
	}
	else {
		return GAMEPLAY::GET_ONSCREEN_KEYBOARD_RESULT();
	}
}

void processSampling() {
	std::ofstream record("sampledCoords.csv");
	WAIT(200);
	Menu hintLine("Press 5 to sample a point. Press 0 to exit.", {}, {});
	while (true) {
		clearArea();
		hintLine.drawVertical(0);
		std::string coords = showPlayerCoords();
		if (IsKeyJustUp(VK_NUMPAD5)) {
			std::string label = getTextInput();
			if (label != "")
				record << label << "," << coords << std::endl;
		}
		if (IsKeyJustUp(VK_NUMPAD0)) {
			record.close();
			WAIT(0);
			break;
		}
		WAIT(0);
	}
}

void processRecording(std::vector<std::string> fileName, std::function<bool()> delegate, int waitTime) {

	std::unordered_map<Entity, int> IDMap;

	std::string dirName;
	std::ofstream record;

	for (int i = 0; i < fileName.size() - 1; i++) {
		if (i != 0) {
			dirName += "/";
		}
		dirName += fileName[i];
		CreateDirectory(dirName.c_str(), NULL);
	}
	record.open(dirName + "/" + fileName.back());
	outputDebugMessage("Logging into file " + dirName + "/" + fileName.back());

	int count = 0;
	int carID = 0;
	int pedID = PED_ID_OFFSET; // constant offset
	while (true) {
		bool seeACar = false;
		for (int i = 0; i < createdVehicles.size(); i++) {
			Entity e = createdVehicles[i];
			if (!IDMap.count(e)) {
				IDMap[e] = carID;
				carID++;
			}

			Vector3d coords3D;
			Vector2 coords2D;
			float heading;
			// float speed;
			getEntityMotion(e, &coords3D, &coords2D, &heading, NULL);

			if (coords2D.x < 0) {
				continue;
			}
			seeACar = true;

			record << count << "," << IDMap[e] << "," << coords2D.x << "," << coords2D.y << "," << coords3D.x << "," << coords3D.y << "," << coords3D.z << "," << heading << std::endl;
		}
		if (seeACar) {
			for (int i = 0; i < createdPeds.size(); i++) {
				Entity e = createdPeds[i];
				if (!IDMap.count(e)) {
					IDMap[e] = pedID;
					pedID++;
				}
				Vector3d coords3D;
				Vector2 coords2D;
				float heading;
				//float speed;
				getEntityMotion(e, &coords3D, &coords2D, &heading, NULL);

				if (coords2D.x < 0) {
					continue;
				}

				record << count << "," << IDMap[e] << "," << coords2D.x << "," << coords2D.y << "," << coords3D.x << "," << coords3D.y << "," << coords3D.z << "," << heading << std::endl;
			}
		}
		if (!delegate()) {
			break;
		}
		WAIT(waitTime);
		if (!seeACar && count == 0)
			continue;
		if (!seeACar && count != 0) {
			outputDebugMessage("Car has passed. Break.");
			break;
		}
		count++;
	}
	record.close();
}

void teleport() {

	if (settings.empty()) {
		return;
	}

	Ped p = PLAYER::PLAYER_PED_ID();
	Vector3d coords = settings["centerCoords"];
	ENTITY::SET_ENTITY_COORDS(p, coords.x, coords.y, coords.z + 5.0, 1, 0, 0, 1);
}

void setCars() {

	if (settings["maxTotalCars"] == 0) return;
	outputDebugMessage("set Cars start.");
	float carLength = getModelLength(settings["carModel"].get<std::string>().c_str());
	outputDebugMessage("carLength is " + std::to_string(carLength));

	// randomly distribute cars on each lane
	std::vector<int> numCars(settings["carStartPositions"].size(), 0);
	int totalCars = randomNum((int)settings["minTotalCars"], settings["maxTotalCars"]);
	for (int i = 0; i < totalCars; i++) {
		int lane = randomNum(0, numCars.size() - 1);
		numCars[lane]++;
	}
	outputDebugMessage("Assigned Lanes for " + std::to_string(totalCars) + " cars.");

	for (int i = 0; i < settings["carStartPositions"].size(); i++) {
		outputDebugMessage("Spawn cars on Lane " + std::to_string(i));

		Vector3d start;
		Vector3d front = Vector3d(settings["carStartPositions"][i]["front"]);
		Vector3d back = Vector3d(settings["carStartPositions"][i]["back"]);
		
		float ratio = randomNum(0.0f, 1.0f);
		start = (back - front) * ratio + front;

		ratio = (settings["carInterval"] + carLength) / SYSTEM::VDIST(front.x, front.y, front.z, back.x, back.y, back.z);

		for (int j = 0; j < numCars[i]; j++) {
			outputDebugMessage("Spawn " + std::to_string(j) + "-th car.");
			Vehicle v = spawnAtCoords(settings["carModel"].get<std::string>().c_str(), MODEL_VEH, start, settings["carHeading"]);
			Ped driver = spawnDriver(v, settings["pedModel"].get<std::string>().c_str());

			int goalNum = randomNum(0, settings["carStartPositions"][i]["goals"].size() - 1);
			float speed = randomNum((float)settings["carMinSpeed"], settings["carMaxSpeed"]);
			Vector3d goal = settings["carStartPositions"][i]["goals"][goalNum];

			createTaskSequence(driver, [&]() {
				AI::TASK_VEHICLE_DRIVE_TO_COORD(0, v, goal.x, goal.y, goal.z, speed, 0, GAMEPLAY::GET_HASH_KEY((char *)(settings["carModel"].get<std::string>().c_str())), settings["driveStyle"], settings["carStopRange"], 1.0);
			}, true);

			start = (back - front) * ratio + start;
			WAIT(0);
		}
	}
}

void setWanderPed() {
	std::vector<Vector3d> pedsPosition;

	if (settings["totalWanderPeds"] == 0) return;

	Vector3d origin = Vector3d(settings["wanderPedsGenArea"]["origin"]);
	Vector3d unit1 = Vector3d(settings["wanderPedsGenArea"]["unit1"]);
	Vector3d unit2 = Vector3d(settings["wanderPedsGenArea"]["unit2"]);

	float walkRadius = SYSTEM::VDIST(unit1.x, unit1.y, unit1.z, unit2.x, unit2.y, unit2.z) * 0.7;

	for (int i = 0; i < settings["totalWanderPeds"]; i++) {
		float ratio1 = randomNum(0.0f, 1.0f);
		float ratio2 = randomNum(0.0f, 1.0f);
		Vector3d start = (unit1 - origin) * ratio1 + (unit2 - origin) * ratio2 + origin;
		bool tooClose = false;
		for (int j = 0; j < pedsPosition.size(); j++) {
			if (SYSTEM::VDIST(pedsPosition[j].x, pedsPosition[j].y, pedsPosition[j].z, start.x, start.y, start.z) < 1.0) {
				tooClose = true;
				break;
			}
		}
		if (tooClose) {
			i--;
			continue;
		}
		pedsPosition.push_back(start);
		Ped p = spawnAtCoords(settings["pedModel"].get<std::string>().c_str(), MODEL_PED, pedsPosition.back());

		createTaskSequence(p, [&]() {
			Vector3d centerCoords = settings["centerCoords"];
			AI::TASK_WANDER_IN_AREA(0, centerCoords.x, centerCoords.y, centerCoords.z, walkRadius, 0.0, 0.0);
			//AI::TASK_WANDER_STANDARD(p, 10.0, 10);
		}, false);

		WAIT(0);
	}
}

void setScriptPed(bool onlyOnce = false) {
	std::vector<std::pair<Vector3d, Vector3d>> pedRoutes;

	int totalScriptPeds = onlyOnce ? 1 : settings["totalScriptPeds"];

	for (int i = 0; i < totalScriptPeds; i++) {
		int startArea = randomNum(0, settings["scriptPedsEndPoints"].size() - 1);
		int startCode = randomNum(0, settings["scriptPedsEndPoints"][startArea].size() - 1);
		Vector3d start = Vector3d(settings["scriptPedsEndPoints"][startArea][startCode]);

		int endArea = randomNum(0, settings["scriptPedsEndPoints"].size() - 1);
		int endCode = randomNum(0, settings["scriptPedsEndPoints"][endArea].size() - 1);
		Vector3d end = Vector3d(settings["scriptPedsEndPoints"][endArea][endCode]);

		bool repeated = false;
		for (int j = 0; j < pedRoutes.size(); j++) {
			if (start == pedRoutes[j].first || (end == pedRoutes[j].first && start == pedRoutes[j].second)) {
				repeated = true;
				break;
			}
		}

		if (endArea == startArea || repeated) {
			i--;
			continue;
		}

		pedRoutes.push_back(std::make_pair(start, end));
		Ped p = spawnAtCoords(settings["pedModel"].get<std::string>().c_str(), MODEL_PED, start);

		createTaskSequence(p, [&]() {
			AI::TASK_GO_STRAIGHT_TO_COORD(0, end.x, end.y, end.z, 1.0, 200000, 0.0, 0.2);
		}, false);

		WAIT(0);
	}
}


void toggleCamera() {

	if (settings.empty()) {
		return;
	}

	static bool initialized(false);
	static bool active(false);

	static Camera customCam = CAM::CREATE_CAM("DEFAULT_SCRIPTED_CAMERA", true);

	if (!initialized) {
		initialized = true;
		CAM::SET_CAM_FOV(customCam, 130.0);
		// CAM::POINT_CAM_AT_COORD(customCam, centerCoords.x, centerCoords.y, centerCoords.z);
		CAM::SET_CINEMATIC_MODE_ACTIVE(false);
	}
	Vector3d centerCoords(settings["centerCoords"]);
	CAM::SET_CAM_COORD(customCam, centerCoords.x, centerCoords.y, centerCoords.z + settings["cameraHeight"]);
	CAM::SET_CAM_ROT(customCam, 270.0, 360.0 - (float)settings["carHeading"], 0.0, 2);

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

void startSimulation() {

	if (settings.empty()) {
		return;
	}

	DWORD nextTickCount;
	bool isPedInjured;
	bool forceBreak;

	auto delegate = [&]() {
		clearArea();

		if (settings["continuousPedGen"]) {
			if (GetTickCount() >= nextTickCount) {
				setScriptPed(true);
				nextTickCount = GetTickCount() + randomNum((int)settings["minPedGenInterval"], settings["maxPedGenInterval"]);
			}
		}

		for (int i = 0; i < createdVehicles.size(); i++) {
			if (PED::IS_PED_INJURED(createdDrivers[i])) {
				outputDebugMessage("Ped injured. Break.");
				isPedInjured = true;
				return false;
			}
			if (AI::GET_SEQUENCE_PROGRESS(createdDrivers[i]) == -1) {
				outputDebugMessage("Deleting Car " + std::to_string(i));
				deleteCar(i);
				i--;
			}
		}

		for (int i = 0; i < createdPeds.size(); i++) {
			if (PED::IS_PED_INJURED(createdPeds[i])) {
				outputDebugMessage("Ped injured. Break.");
				isPedInjured = true;
				return false;
			}
			if (AI::GET_SEQUENCE_PROGRESS(createdPeds[i]) == -1) {
				outputDebugMessage("Deleting Ped " + std::to_string(i));
				deletePed(i);
				i--;
			}
		}
		if (IsKeyJustUp(VK_NUMPAD0)) {
			outputDebugMessage("Force break.");
			forceBreak = true;
			return false;
		}

		if (createdVehicles.size() == 0) {
			outputDebugMessage("No vehicles. Stop.");
			return false;
		}
		return true;
	};


	std::string curType = "straight";

	std::string curConfig = "_nw_250";

	for (int i = 0; i < 100; i++) {
		if (true) {
			outputDebugMessage("start simulation " + std::to_string(i) + ".");
			clearArea();
			outputDebugMessage("clearArea complete.");
			setCars();
			outputDebugMessage("setCars complete.");
			setWanderPed();
			outputDebugMessage("setWanderPed complete.");
			setScriptPed();
			outputDebugMessage("setScriptPed complete.");
			nextTickCount = GetTickCount();
			isPedInjured = false;
			forceBreak = false;
			json recordFile = settings["recordDirectory"];
			recordFile.push_back("record" + std::to_string(i));
			processRecording(recordFile, delegate, settings["period"]);
			outputDebugMessage("setScriptPed complete.");
			deleteAllCreated();
			outputDebugMessage("deleteAll complete.");
			WAIT(0);
			if (isPedInjured) {
				i--;
			}
			if (forceBreak) {
				break;
			}
		}
		else {
			int j = randomNum(0, 5);
			std::string fileName = "RecordSettings/settings-" + curType + std::to_string(j) + curConfig + ".json";

			std::ifstream config(fileName);
			outputDebugMessage("Loading config file " + fileName);
			settings = json();
			config >> settings;
			config.close();
			settingFile = fileName;

			outputDebugMessage("start simulation " + std::to_string(i));

			teleport();
			ENTITY::SET_ENTITY_COLLISION(PLAYER::PLAYER_PED_ID(), 0, 0);
			PED::SET_PED_GRAVITY(PLAYER::PLAYER_PED_ID(), 0);

			toggleCamera();
			toggleCamera();
			clearArea();
			outputDebugMessage("clearArea complete.");
			setCars();
			outputDebugMessage("setCars complete.");
			setWanderPed();
			outputDebugMessage("setWanderPed complete.");
			setScriptPed();
			outputDebugMessage("setScriptPed complete.");
			nextTickCount = GetTickCount();
			isPedInjured = false;
			forceBreak = false;
			std::vector<std::string> recordFile = { "[Mix]" + curType + curConfig };
			recordFile.push_back("[" + std::to_string(j) + "]" + "record" + std::to_string(i));
			processRecording(recordFile, delegate, settings["period"]);
			outputDebugMessage("recoding complete.");
			deleteAllCreated();
			outputDebugMessage("deleteAll complete.");
			WAIT(0);
			if (isPedInjured) {
				i--;
			}
			if (forceBreak) {
				break;
			}
		}
	}


}


void loadSettings() {
	auto fileList = readAllFiles(settingsDirectory);

	auto loadFile = [&](int i) {
		std::ifstream config(fileList[i]);
		outputDebugMessage("Load config file" + fileList[i]);
		if (config.is_open()) {
			settings = json();
			config >> settings;
			config.close();
			settingFile = fileList[i];
		}
	};

	std::vector<std::function<void()>> funcList;

	for (int i = 0; i < fileList.size(); i++) {
		funcList.push_back(std::bind(loadFile, i));
	}

	Menu loadSettingsMenu("Choose Setting File", fileList, funcList);

	loadSettingsMenu.processMenu();

}

void recordPosition(json& saveSpot, bool appendBack = false, bool getHeading = false) {
	Menu hintLine("Press 5 to record coords. Press 0 to exit.", {}, {});
	Menu SuccessMessage("Point recorded", {}, {});
	while (true) {
		clearArea();
		hintLine.drawVertical(0);
		if (IsKeyJustUp(VK_NUMPAD5)) {
			if (!getHeading) {
				Vector3d centerCoords;
				getEntityMotion(PLAYER::PLAYER_PED_ID(), &centerCoords);
				json coords_j = { centerCoords.x, centerCoords.y, centerCoords.z };
				if (appendBack == false) {
					saveSpot = coords_j;
				}
				else {
					saveSpot.push_back(coords_j);
				}
			}
			else {
				float heading;
				getEntityMotion(PLAYER::PLAYER_PED_ID(), NULL, NULL, &heading);
				saveSpot = heading;
			}
		}
		if (IsKeyJustUp(VK_NUMPAD0)) {
			WAIT(0);
			break;
		}
		WAIT(0);
	}
}

void saveSettings() {
	std::string fileName = getTextInput();
	if (fileName == "") {
		fileName = settingFile;
	}
	else {
		settingFile = settingsDirectory + "/" + fileName;
	}
	std::ofstream config(settingFile);
	if (config.is_open()) {
		outputDebugMessage("Save to file " + fileName);
		config << settings;
		config.close();
	}
}

void setCenterCoords() {
	recordPosition(settings["centerCoords"]);
}

void addCarStartPosition() {
	json newPosition;
	printOnScreen("Record front of lane");
	recordPosition(newPosition["front"]);
	printOnScreen("Record back of lane");
	recordPosition(newPosition["back"]);
	printOnScreen("Record goals of lane.");
	recordPosition(newPosition["goals"], true);
	settings["carStartPositions"].push_back(newPosition);
}

void clearCarStartPositions() {
	settings["carStartPositions"] = json();
}

void setCarHeading() {
	recordPosition(settings["carHeading"], false, true);
}

void setWanderPedsGenArea() {
	printOnScreen("Record center corner of area.");
	recordPosition(settings["wanderPedsGenArea"]["origin"]);
	printOnScreen("Record side corner 1 of area.");
	recordPosition(settings["wanderPedsGenArea"]["unit1"]);
	printOnScreen("Record side corner 2 of area.");
	recordPosition(settings["wanderPedsGenArea"]["unit2"]);
}

void addScriptPedsEndPoints() {
	json oneSet;
	printOnScreen("Record points on one corner of sidewalk.");
	recordPosition(oneSet, true);
	if (!oneSet.empty()) {
		settings["scriptPedsEndPoints"].push_back(oneSet);
	}
}

void clearScriptPedsEndPoints() {
	settings["scriptPedsEndPoints"] = json();
}

void modifySettings() {

	if (settings.empty()) {
		loadSettings();
	}

	Menu modifySettingsMenu("Modify Settings",
		{ "Save Settings to File",
			"Center Coords",
			"Add carStartPosition",
			"Clear carStartPositions",
			"Car Heading at Spawn",
			"wanderPedsGenArea",
			"Add One Set of scriptPedsEndPoints",
			"Clear scriptPedsEndPoints"
		},
		{ saveSettings,
			setCenterCoords,
			addCarStartPosition,
			clearCarStartPositions,
			setCarHeading,
			setWanderPedsGenArea,
			addScriptPedsEndPoints,
			clearScriptPedsEndPoints
		});

	modifySettingsMenu.processMenu();
}

void loadPredictions(std::unordered_map<int, std::unordered_map <int, std::vector<Vector2>>>& coordsMap) {
	std::ifstream prediction("record2.csv");
	int timestamp;
	while (prediction >> timestamp) {
		prediction.ignore(1000, ',');

		int id;
		Vector3d coords;

		prediction >> id;
		prediction.ignore(1000, ',');

		prediction >> coords.x;
		prediction.ignore(1000, ',');
		prediction >> coords.y;
		prediction.ignore(1000, ',');

		float searchHeight = 500;

		GAMEPLAY::GET_GROUND_Z_FOR_3D_COORD(coords.x, coords.y, searchHeight, &(coords.z) , 0);

		Vector2 coords2D;
		GRAPHICS::_WORLD3D_TO_SCREEN2D(coords.x, coords.y, coords.z, &(coords2D.x), &(coords2D.y));
		int screen_w, screen_h;
		GRAPHICS::GET_SCREEN_RESOLUTION(&screen_w, &screen_h);
		coords2D.x *= screen_w;
		coords2D.y *= screen_h;

		prediction.ignore(1000, '\n');
		if (id < PED_ID_OFFSET) {
			coordsMap[timestamp][id].push_back(coords2D);
		}
	}
	prediction.close();
}

void replay() {
	if (settings.empty()) {
		return;
	}

	int color[][3] = { {255,0,0}, {0,255,0}, {0,0,255} };

	std::ifstream record("record1.csv");

	std::unordered_map<int, std::pair<Entity, bool>> idMap;
	std::unordered_map<int, std::unordered_map <int, std::vector<Vector2>>> coordsMap;
	loadPredictions(coordsMap);

	int waitTime = 0;
	int now = -1;

	int timestamp;

	while (record >> timestamp) {
		clearArea();

		record.ignore(1000, ',');

		int id;
		Vector3d coords3D;
		float heading;
		//float speed;

		record >> id;
		record.ignore(1000, ',');

		// ignore 2D coords, not needed
		record.ignore(1000, ',');
		record.ignore(1000, ',');

		record >> coords3D.x;
		record.ignore(1000, ',');
		record >> coords3D.y;
		record.ignore(1000, ',');
		record >> coords3D.z;
		record.ignore(1000, ',');

		record >> heading;
		//record.ignore(1000, ',');
		//record >> speed;
		record.ignore(1000, '\n');


		if (timestamp != now) {
			now = timestamp;
			for (auto& pair : idMap) {
				bool& visited = pair.second.second;
				if (visited == true) { // reset visted value
					visited = false;
				}
				else { // delete the unvisited entity
					int id = pair.first;
					Entity e = pair.second.first;
					if (id >= PED_ID_OFFSET) {
						for (int i = 0; i < createdPeds.size(); i++) {
							if (createdPeds[i] == e) {
								deletePed(i);
								break;
							}
						}
					}
					else {
						for (int i = 0; i < createdVehicles.size(); i++) {
							if (createdVehicles[i] == e) {
								deleteCar(i);
								break;
							}
						}
					}
					idMap.erase(id);
				}
			}
			waitTime = settings["period"];
		}
		else {
			waitTime = 0;
		}

		if (id >= PED_ID_OFFSET) {
			if (!idMap.count(id)) {
				Ped p = spawnAtCoords(settings["pedModel"].get<std::string>().c_str(), MODEL_PED, coords3D, heading);
				idMap[id] = std::make_pair(p, true);
			}
			else {
				Ped p = idMap[id].first;
				idMap[id].second = true;
				// AI::TASK_GO_STRAIGHT_TO_COORD(p, coords3D.x, coords3D.y, coords3D.z, speed, 2000000, heading, 0.2);
				ENTITY::SET_ENTITY_COORDS_NO_OFFSET(p, coords3D.x, coords3D.y, coords3D.z, 1, 0, 0);
				ENTITY::SET_ENTITY_HEADING(p, heading);
			}
		}
		else {
			if (!idMap.count(id)) {
				Vehicle v = spawnAtCoords(settings["carModel"].get<std::string>().c_str(), MODEL_VEH, coords3D, heading);
				spawnDriver(v, settings["pedModel"].get<std::string>().c_str());
				idMap[id] = std::make_pair(v, true);
			}
			else {
				Vehicle v = idMap[id].first;
				idMap[id].second = true;

				int i;
				for (i = 0; i < createdVehicles.size(); i++) {
					if (createdVehicles[i] == v) {
						break;
					}
				}
				Ped driver = createdDrivers[i];

				ENTITY::SET_ENTITY_COORDS_NO_OFFSET(v, coords3D.x, coords3D.y, coords3D.z, 1, 0, 0);
				ENTITY::SET_ENTITY_HEADING(v, heading);

				// AI::TASK_VEHICLE_DRIVE_TO_COORD(driver, v, coords3D.x, coords3D.y, coords3D.z, speed, 0, GAMEPLAY::GET_HASH_KEY((char *)(settings["carModel"].get<std::string>().c_str())), 786475, 0.2, 1.0);
			}

		}
		if (coordsMap.count(timestamp)) {
			if (waitTime != 0) {
				DWORD maxTickCount = GetTickCount() + waitTime;
				while (GetTickCount() < maxTickCount) {
					for (auto& p : coordsMap[timestamp]) {
						for (auto c : p.second) {
							draw_mark_at(c, color[p.first][0], color[p.first][1], color[p.first][2]);
						}
					}
					WAIT(0);
				}
			}
			else {
				for (auto& p : coordsMap[timestamp]) {
					for (auto c : p.second) {
						draw_mark_at(c, color[p.first][0], color[p.first][1], color[p.first][2]);
					}
				}
			}
		}
		else {
			if (waitTime != 0) {
				WAIT(waitTime);
			}
		}
	}

	record.close();
	deleteAllCreated();

	now++;
	while (coordsMap.count(now)) {
		DWORD maxTickCount = GetTickCount() + settings["period"];
		clearArea();
		while (GetTickCount() < maxTickCount) {
			for (auto& p : coordsMap[now]) {
				for (auto c : p.second) {
					draw_mark_at(c, color[p.first][0], color[p.first][1], color[p.first][2]);
				}
			}
			WAIT(0);
		}
		now++;
	}

}

void main() {
	Menu mainMenu("Main Menu",
		{ "Sample Points",
			"Teleport to Scene",
			"Toggle Camera",
			"Start Simulation",
			"Modify Settings",
			"Load Settings",
			"Replay",
			"Reset Player"
		},
		{ processSampling,
			teleport,
			toggleCamera,
			startSimulation,
			modifySettings,
			loadSettings,
			replay,
			resetPlayer
		});

	remove("debug.txt");

	while (true) {
		if (IsKeyJustUp(VK_F6)) {
			AUDIO::PLAY_SOUND_FRONTEND(-1, "NAV_UP_DOWN", "HUD_FRONTEND_DEFAULT_SOUNDSET", 0);
			mainMenu.processMenu();
		}
		WAIT(0);
	}
}

void ScriptMain() {
	srand(GetTickCount());
	main();
}
