#include "graphics.h"

static unsigned char colormap[COLORNUM * 3];

void draw_rect(float A_0, float A_1, float A_2, float A_3, int A_4, int A_5, int A_6, int A_7)
{
	GRAPHICS::DRAW_RECT((A_0 + (A_2 * 0.5f)), (A_1 + (A_3 * 0.5f)), A_2, A_3, A_4, A_5, A_6, A_7);
}

void draw_menu_line(std::string caption, float lineWidth, float lineHeight, float lineTop, float lineLeft, float textLeft, bool active, bool title, bool rescaleText)
{
	// default values
	int text_col[4] = { 255, 255, 255, 255 },
		rect_col[4] = { 70, 95, 95, 255 };
	float text_scale = 0.35;
	int font = 0;

	// correcting values for active line
	if (active)
	{
		text_col[0] = 0;
		text_col[1] = 0;
		text_col[2] = 0;

		rect_col[0] = 218;
		rect_col[1] = 242;
		rect_col[2] = 216;

		if (rescaleText) text_scale = 0.40;
	}

	if (title)
	{
		rect_col[0] = 0;
		rect_col[1] = 0;
		rect_col[2] = 0;

		if (rescaleText) text_scale = 0.50;
		font = 1;
	}

	int screen_w, screen_h;
	GRAPHICS::GET_SCREEN_RESOLUTION(&screen_w, &screen_h);

	textLeft += lineLeft;

	float lineWidthScaled = lineWidth / (float)screen_w; // line width
	float lineTopScaled = lineTop / (float)screen_h; // line top offset
	float textLeftScaled = textLeft / (float)screen_w; // text left offset
	float lineHeightScaled = lineHeight / (float)screen_h; // line height

	float lineLeftScaled = lineLeft / (float)screen_w;

	// this is how it's done in original scripts

	// text upper part
	UI::SET_TEXT_FONT(font);
	UI::SET_TEXT_SCALE(0.0, text_scale);
	UI::SET_TEXT_COLOUR(text_col[0], text_col[1], text_col[2], text_col[3]);
	UI::SET_TEXT_CENTRE(0);
	UI::SET_TEXT_DROPSHADOW(0, 0, 0, 0, 0);
	UI::SET_TEXT_EDGE(0, 0, 0, 0, 0);
	UI::_SET_TEXT_ENTRY("STRING");
	UI::_ADD_TEXT_COMPONENT_STRING((LPSTR)caption.c_str());
	UI::_DRAW_TEXT(textLeftScaled, (((lineTopScaled + 0.00278f) + lineHeightScaled) - 0.005f));

	// text lower part
	UI::SET_TEXT_FONT(font);
	UI::SET_TEXT_SCALE(0.0, text_scale);
	UI::SET_TEXT_COLOUR(text_col[0], text_col[1], text_col[2], text_col[3]);
	UI::SET_TEXT_CENTRE(0);
	UI::SET_TEXT_DROPSHADOW(0, 0, 0, 0, 0);
	UI::SET_TEXT_EDGE(0, 0, 0, 0, 0);
	UI::_SET_TEXT_GXT_ENTRY("STRING");
	UI::_ADD_TEXT_COMPONENT_STRING((LPSTR)caption.c_str());
	int num25 = UI::_0x9040DFB09BE75706(textLeftScaled, (((lineTopScaled + 0.00278f) + lineHeightScaled) - 0.005f));

	// rect
	draw_rect(lineLeftScaled, lineTopScaled + (0.00278f),
		lineWidthScaled, ((((float)(num25)* UI::_0xDB88A37483346780(text_scale, 0)) + (lineHeightScaled * 2.0f)) + 0.005f),
		rect_col[0], rect_col[1], rect_col[2], rect_col[3]);
}

void printOnScreen(std::string s) {
	DWORD maxTickCount = GetTickCount() + 5000;
	while (GetTickCount() < maxTickCount) {
		draw_menu_line(s, 350.0, 15.0, 15, 400, 5.0, false, true);
		WAIT(0);
	}
	WAIT(0);
}

void draw_mark_at(Vector2 coords, int r, int g, int b)
{
	float sideLen = 10;

	int screen_w, screen_h;
	GRAPHICS::GET_SCREEN_RESOLUTION(&screen_w, &screen_h);

	GRAPHICS::DRAW_RECT((coords.x - sideLen/2)/screen_w, (coords.y - sideLen/2)/screen_h,
		sideLen/screen_w, sideLen/screen_h, r, g, b, 255);
}

void draw_line_positive(Vector3 initV, float unit_x_length, float unit_bar_length, int num_interp, float k, int colorOffset)
{
	for (int i = 0; i < num_interp; i++)
	{
		Vector3 currentV, nextV;	// relative
		currentV.x = unit_x_length * i; currentV.y = k * currentV.x; currentV.z = 0;
		nextV.x = unit_x_length * (i + 1); nextV.y = k * nextV.x; nextV.z = 0;
		currentV.x += initV.x; currentV.y += initV.y; currentV.z += initV.z;
		nextV.x += initV.x; nextV.y += initV.y; nextV.z += initV.z;
		float theta = atan(k);
		for (int j = colorOffset; j < COLORNUM; j++)
		{
			int drawPos = j - colorOffset;
			GRAPHICS::DRAW_POLY(nextV.x + unit_bar_length * drawPos*sin(theta), nextV.y - unit_bar_length * drawPos*cos(theta), nextV.z, currentV.x + unit_bar_length * drawPos*sin(theta), currentV.y - unit_bar_length * drawPos*cos(theta), currentV.z, currentV.x + unit_bar_length * (drawPos + 1)*sin(theta), currentV.y - unit_bar_length * (drawPos + 1)*cos(theta), currentV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
			GRAPHICS::DRAW_POLY(currentV.x + unit_bar_length * (drawPos + 1)*sin(theta), currentV.y - unit_bar_length * (drawPos + 1)*cos(theta), currentV.z, nextV.x + unit_bar_length * (drawPos + 1)*sin(theta), nextV.y - unit_bar_length * (drawPos + 1)*cos(theta), nextV.z, nextV.x + unit_bar_length * drawPos*sin(theta), nextV.y - unit_bar_length * drawPos*cos(theta), nextV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
			GRAPHICS::DRAW_POLY(currentV.x - unit_bar_length * drawPos*sin(theta), currentV.y + unit_bar_length * drawPos*cos(theta), currentV.z, nextV.x - unit_bar_length * drawPos*sin(theta), nextV.y + unit_bar_length * drawPos*cos(theta), nextV.z, nextV.x - unit_bar_length * (drawPos + 1)*sin(theta), nextV.y + unit_bar_length * (drawPos + 1)*cos(theta), nextV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
			GRAPHICS::DRAW_POLY(nextV.x - unit_bar_length * (drawPos + 1)*sin(theta), nextV.y + unit_bar_length * (drawPos + 1)*cos(theta), nextV.z, currentV.x - unit_bar_length * (drawPos + 1)*sin(theta), currentV.y + unit_bar_length * (drawPos + 1)*cos(theta), currentV.z, currentV.x - unit_bar_length * drawPos*sin(theta), currentV.y + unit_bar_length * drawPos*cos(theta), currentV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
		}
	}
}

void draw_line_negative(Vector3 initV, float unit_x_length, float unit_bar_length, int num_interp, float k, int colorOffset)
{
	for (int i = num_interp - 1; i >= 0; i--)
	{
		Vector3 currentV, nextV;	// relative
		currentV.x = unit_x_length * (i + 1); currentV.y = k * currentV.x; currentV.z = 0;
		nextV.x = unit_x_length * i; nextV.y = k * nextV.x; nextV.z = 0;
		currentV.x += initV.x; currentV.y += initV.y; currentV.z += initV.z;
		nextV.x += initV.x; nextV.y += initV.y; nextV.z += initV.z;
		float theta = atan(k);
		for (int j = colorOffset; j < COLORNUM; j++)
		{
			int drawPos = j - colorOffset;
			GRAPHICS::DRAW_POLY(nextV.x + unit_bar_length * drawPos*sin(theta), nextV.y - unit_bar_length * drawPos*cos(theta), nextV.z, currentV.x + unit_bar_length * drawPos*sin(theta), currentV.y - unit_bar_length * drawPos*cos(theta), currentV.z, currentV.x + unit_bar_length * (drawPos + 1)*sin(theta), currentV.y - unit_bar_length * (drawPos + 1)*cos(theta), currentV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
			GRAPHICS::DRAW_POLY(currentV.x + unit_bar_length * (drawPos + 1)*sin(theta), currentV.y - unit_bar_length * (drawPos + 1)*cos(theta), currentV.z, nextV.x + unit_bar_length * (drawPos + 1)*sin(theta), nextV.y - unit_bar_length * (drawPos + 1)*cos(theta), nextV.z, nextV.x + unit_bar_length * drawPos*sin(theta), nextV.y - unit_bar_length * drawPos*cos(theta), nextV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
			GRAPHICS::DRAW_POLY(currentV.x - unit_bar_length * drawPos*sin(theta), currentV.y + unit_bar_length * drawPos*cos(theta), currentV.z, nextV.x - unit_bar_length * drawPos*sin(theta), nextV.y + unit_bar_length * drawPos*cos(theta), nextV.z, nextV.x - unit_bar_length * (drawPos + 1)*sin(theta), nextV.y + unit_bar_length * (drawPos + 1)*cos(theta), nextV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
			GRAPHICS::DRAW_POLY(nextV.x - unit_bar_length * (drawPos + 1)*sin(theta), nextV.y + unit_bar_length * (drawPos + 1)*cos(theta), nextV.z, currentV.x - unit_bar_length * (drawPos + 1)*sin(theta), currentV.y + unit_bar_length * (drawPos + 1)*cos(theta), currentV.z, currentV.x - unit_bar_length * drawPos*sin(theta), currentV.y + unit_bar_length * drawPos*cos(theta), currentV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
		}
	}
}

void draw_paraboa_positive_x(Vector3 initV, float unit_x_length, float unit_bar_length, int num_interp, float a, float b, int colorOffset)
{
	for (int i = 0; i < num_interp; i++)
	{
		Vector3 currentV, nextV;	// relative
		currentV.x = unit_x_length * i; currentV.y = a * currentV.x*currentV.x + b * currentV.x; currentV.z = 0;
		nextV.x = unit_x_length * (i + 1); nextV.y = a * nextV.x*nextV.x + b * nextV.x; nextV.z = 0;
		currentV.x += initV.x; currentV.y += initV.y; currentV.z += initV.z;
		nextV.x += initV.x; nextV.y += initV.y; nextV.z += initV.z;
		float theta = atan((nextV.y - currentV.y) / (nextV.x - currentV.x));
		for (int j = colorOffset; j < COLORNUM; j++)
		{
			GRAPHICS::DRAW_POLY(nextV.x + unit_bar_length * j*sin(theta), nextV.y - unit_bar_length * j*cos(theta), nextV.z, currentV.x + unit_bar_length * j*sin(theta), currentV.y - unit_bar_length * j*cos(theta), currentV.z, currentV.x + unit_bar_length * (j + 1)*sin(theta), currentV.y - unit_bar_length * (j + 1)*cos(theta), currentV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
			GRAPHICS::DRAW_POLY(currentV.x + unit_bar_length * (j + 1)*sin(theta), currentV.y - unit_bar_length * (j + 1)*cos(theta), currentV.z, nextV.x + unit_bar_length * (j + 1)*sin(theta), nextV.y - unit_bar_length * (j + 1)*cos(theta), nextV.z, nextV.x + unit_bar_length * j*sin(theta), nextV.y - unit_bar_length * j*cos(theta), nextV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
			GRAPHICS::DRAW_POLY(currentV.x - unit_bar_length * j*sin(theta), currentV.y + unit_bar_length * j*cos(theta), currentV.z, nextV.x - unit_bar_length * j*sin(theta), nextV.y + unit_bar_length * j*cos(theta), nextV.z, nextV.x - unit_bar_length * (j + 1)*sin(theta), nextV.y + unit_bar_length * (j + 1)*cos(theta), nextV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
			GRAPHICS::DRAW_POLY(nextV.x - unit_bar_length * (j + 1)*sin(theta), nextV.y + unit_bar_length * (j + 1)*cos(theta), nextV.z, currentV.x - unit_bar_length * (j + 1)*sin(theta), currentV.y + unit_bar_length * (j + 1)*cos(theta), currentV.z, currentV.x - unit_bar_length * j*sin(theta), currentV.y + unit_bar_length * j*cos(theta), currentV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
		}
	}
}

void draw_paraboa_negative_x(Vector3 initV, float unit_x_length, float unit_bar_length, int num_interp, float a, float b, int colorOffset)
{
	for (int i = num_interp - 1; i >= 0; i--)
	{
		Vector3 currentV, nextV;	// relative
		currentV.x = unit_x_length * (i + 1); currentV.y = a * currentV.x*currentV.x + b * currentV.x; currentV.z = 0;
		nextV.x = unit_x_length * i; nextV.y = a * nextV.x*nextV.x + b * nextV.x; nextV.z = 0;
		currentV.x += initV.x; currentV.y += initV.y; currentV.z += initV.z;
		nextV.x += initV.x; nextV.y += initV.y; nextV.z += initV.z;
		float theta = atan((nextV.y - currentV.y) / (nextV.x - currentV.x));
		for (int j = colorOffset; j < COLORNUM; j++)
		{
			GRAPHICS::DRAW_POLY(nextV.x + unit_bar_length * j*sin(theta), nextV.y - unit_bar_length * j*cos(theta), nextV.z, currentV.x + unit_bar_length * j*sin(theta), currentV.y - unit_bar_length * j*cos(theta), currentV.z, currentV.x + unit_bar_length * (j + 1)*sin(theta), currentV.y - unit_bar_length * (j + 1)*cos(theta), currentV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
			GRAPHICS::DRAW_POLY(currentV.x + unit_bar_length * (j + 1)*sin(theta), currentV.y - unit_bar_length * (j + 1)*cos(theta), currentV.z, nextV.x + unit_bar_length * (j + 1)*sin(theta), nextV.y - unit_bar_length * (j + 1)*cos(theta), nextV.z, nextV.x + unit_bar_length * j*sin(theta), nextV.y - unit_bar_length * j*cos(theta), nextV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
			GRAPHICS::DRAW_POLY(currentV.x - unit_bar_length * j*sin(theta), currentV.y + unit_bar_length * j*cos(theta), currentV.z, nextV.x - unit_bar_length * j*sin(theta), nextV.y + unit_bar_length * j*cos(theta), nextV.z, nextV.x - unit_bar_length * (j + 1)*sin(theta), nextV.y + unit_bar_length * (j + 1)*cos(theta), nextV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
			GRAPHICS::DRAW_POLY(nextV.x - unit_bar_length * (j + 1)*sin(theta), nextV.y + unit_bar_length * (j + 1)*cos(theta), nextV.z, currentV.x - unit_bar_length * (j + 1)*sin(theta), currentV.y + unit_bar_length * (j + 1)*cos(theta), currentV.z, currentV.x - unit_bar_length * j*sin(theta), currentV.y + unit_bar_length * j*cos(theta), currentV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
		}
	}
}

void draw_paraboa_positive_y(Vector3 initV, float unit_y_length, float unit_bar_length, int num_interp, float a, float b, int colorOffset)
{
	for (int i = 0; i < num_interp; i++)
	{
		Vector3 currentV, nextV;
		currentV.y = unit_y_length * i; currentV.x = a * currentV.y*currentV.y + b * currentV.y; currentV.z = 0;
		nextV.y = unit_y_length * (i + 1); nextV.x = a * nextV.y*nextV.y + b * nextV.y; nextV.z = 0;
		currentV.x += initV.x; currentV.y += initV.y; currentV.z += initV.z;
		nextV.x += initV.x; nextV.y += initV.y; nextV.z += initV.z;
		float theta = atan((nextV.x - currentV.x) / (nextV.y - currentV.y));
		for (int j = colorOffset; j < COLORNUM; j++)
		{
			GRAPHICS::DRAW_POLY(nextV.x + unit_bar_length * j*cos(theta), nextV.y - unit_bar_length * j*sin(theta), nextV.z, currentV.x + unit_bar_length * j*cos(theta), currentV.y - unit_bar_length * j*sin(theta), currentV.z, currentV.x + unit_bar_length * (j + 1)*cos(theta), currentV.y - unit_bar_length * (j + 1)*sin(theta), currentV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
			GRAPHICS::DRAW_POLY(currentV.x + unit_bar_length * (j + 1)*cos(theta), currentV.y - unit_bar_length * (j + 1)*sin(theta), currentV.z, nextV.x + unit_bar_length * (j + 1)*cos(theta), nextV.y - unit_bar_length * (j + 1)*sin(theta), nextV.z, nextV.x + unit_bar_length * j*cos(theta), nextV.y - unit_bar_length * j*sin(theta), nextV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
			GRAPHICS::DRAW_POLY(currentV.x - unit_bar_length * j*cos(theta), currentV.y + unit_bar_length * j*sin(theta), currentV.z, nextV.x - unit_bar_length * j*cos(theta), nextV.y + unit_bar_length * j*sin(theta), nextV.z, nextV.x - unit_bar_length * (j + 1)*cos(theta), nextV.y + unit_bar_length * (j + 1)*sin(theta), nextV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
			GRAPHICS::DRAW_POLY(nextV.x - unit_bar_length * (j + 1)*cos(theta), nextV.y + unit_bar_length * (j + 1)*sin(theta), nextV.z, currentV.x - unit_bar_length * (j + 1)*cos(theta), currentV.y + unit_bar_length * (j + 1)*sin(theta), currentV.z, currentV.x - unit_bar_length * j*cos(theta), currentV.y + unit_bar_length * j*sin(theta), currentV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
		}
	}
}

void draw_paraboa_negative_y(Vector3 initV, float unit_y_length, float unit_bar_length, int num_interp, float a, float b, int colorOffset)
{
	for (int i = num_interp - 1; i >= 0; i--)
	{
		Vector3 currentV, nextV;
		currentV.y = unit_y_length * (i + 1); currentV.x = a * currentV.y*currentV.y + b * currentV.y; currentV.z = 0;
		nextV.y = unit_y_length * i; nextV.x = a * nextV.y*nextV.y + b * nextV.y; nextV.z = 0;
		currentV.x += initV.x; currentV.y += initV.y; currentV.z += initV.z;
		nextV.x += initV.x; nextV.y += initV.y; nextV.z += initV.z;
		float theta = atan((nextV.x - currentV.x) / (nextV.y - currentV.y));
		for (int j = colorOffset; j < COLORNUM; j++)
		{
			GRAPHICS::DRAW_POLY(nextV.x + unit_bar_length * j*cos(theta), nextV.y - unit_bar_length * j*sin(theta), nextV.z, currentV.x + unit_bar_length * j*cos(theta), currentV.y - unit_bar_length * j*sin(theta), currentV.z, currentV.x + unit_bar_length * (j + 1)*cos(theta), currentV.y - unit_bar_length * (j + 1)*sin(theta), currentV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
			GRAPHICS::DRAW_POLY(currentV.x + unit_bar_length * (j + 1)*cos(theta), currentV.y - unit_bar_length * (j + 1)*sin(theta), currentV.z, nextV.x + unit_bar_length * (j + 1)*cos(theta), nextV.y - unit_bar_length * (j + 1)*sin(theta), nextV.z, nextV.x + unit_bar_length * j*cos(theta), nextV.y - unit_bar_length * j*sin(theta), nextV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
			GRAPHICS::DRAW_POLY(currentV.x - unit_bar_length * j*cos(theta), currentV.y + unit_bar_length * j*sin(theta), currentV.z, nextV.x - unit_bar_length * j*cos(theta), nextV.y + unit_bar_length * j*sin(theta), nextV.z, nextV.x - unit_bar_length * (j + 1)*cos(theta), nextV.y + unit_bar_length * (j + 1)*sin(theta), nextV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
			GRAPHICS::DRAW_POLY(nextV.x - unit_bar_length * (j + 1)*cos(theta), nextV.y + unit_bar_length * (j + 1)*sin(theta), nextV.z, currentV.x - unit_bar_length * (j + 1)*cos(theta), currentV.y + unit_bar_length * (j + 1)*sin(theta), currentV.z, currentV.x - unit_bar_length * j*cos(theta), currentV.y + unit_bar_length * j*sin(theta), currentV.z, colormap[3 * j], colormap[3 * j + 1], colormap[3 * j + 2], 100);
		}
	}
}

void draw_quadrant(Vector3 forwardVec, float x0, float y0, Vector3 initV, int num_interp, int colorOffset)
{
	if (forwardVec.x*x0 < 0 && forwardVec.y*y0 < 0)
		return;

	float unit_x_length = x0 / num_interp;
	float unit_y_length = y0 / num_interp;
	float unit_bar_length = .002;
	if (forwardVec.x * x0 < 0 && forwardVec.y * y0 > 0)
	{
		float a = -y0 / (x0*x0);
		float b = 2 * y0 / x0;
		if (x0 > 0)
			draw_paraboa_positive_x(initV, unit_x_length, unit_bar_length, num_interp, a, b, colorOffset);
		else
			draw_paraboa_negative_x(initV, unit_x_length, unit_bar_length, num_interp, a, b, colorOffset);
	}

	if (forwardVec.x * x0 > 0 && forwardVec.y * y0 < 0)
	{
		float a = -x0 / (y0*y0);
		float b = 2 * x0 / y0;
		if (y0 > 0)
			draw_paraboa_positive_y(initV, unit_y_length, unit_bar_length, num_interp, a, b, colorOffset);
		else
			draw_paraboa_negative_y(initV, unit_y_length, unit_bar_length, num_interp, a, b, colorOffset);
	}

	if (forwardVec.x * x0 > 0 && forwardVec.y * y0 > 0)
	{
		float tg_xtype = 2 * y0 / x0;
		float tg_ytype = y0 / (2 * x0);
		float tg_player = forwardVec.y / forwardVec.x;

		if ((tg_player > tg_xtype && tg_player > tg_ytype) || (tg_player < tg_xtype && tg_player < tg_ytype))
		{
			float cos_xtype = (1 + tg_player * tg_xtype) / (sqrt(1 + tg_player * tg_player)*sqrt(1 + tg_xtype * tg_xtype));
			float cos_ytype = (1 + tg_player * tg_ytype) / (sqrt(1 + tg_player * tg_player)*sqrt(1 + tg_ytype * tg_ytype));

			if (cos_xtype > cos_ytype)
			{
				float a = -y0 / (x0*x0);
				float b = 2 * y0 / x0;
				if (x0 > 0)
					draw_paraboa_positive_x(initV, unit_x_length, unit_bar_length, num_interp, a, b, colorOffset);
				else
					draw_paraboa_negative_x(initV, unit_x_length, unit_bar_length, num_interp, a, b, colorOffset);
			}
			else
			{
				float a = -x0 / (y0*y0);
				float b = 2 * x0 / y0;
				if (y0 > 0)
					draw_paraboa_positive_y(initV, unit_y_length, unit_bar_length, num_interp, a, b, colorOffset);
				else
					draw_paraboa_negative_y(initV, unit_y_length, unit_bar_length, num_interp, a, b, colorOffset);
			}
		}
		else
		{
			float k = y0 / x0;
			if (x0 > 0)
				draw_line_positive(initV, unit_x_length, unit_bar_length, num_interp, k, colorOffset);
			else
				draw_line_negative(initV, unit_x_length, unit_bar_length, num_interp, k, colorOffset);
		}
	}
}

void something() {
	int screen_res_x;

	int screen_res_y;

	GRAPHICS::GET_SCREEN_RESOLUTION(&screen_res_x, &screen_res_y);

	char* dword = "commonmenu";

	char* dword2 = "gradient_bgd";

	Vector3 texture_res = GRAPHICS::GET_TEXTURE_RESOLUTION(dword, dword2);

	GRAPHICS::DRAW_SPRITE(dword, dword2, 0.16f, 0.5f, texture_res.x / (float)screen_res_x, texture_res.y / (float)screen_res_y, 0.0f, 255, 255, 255, 255);


}

void load_colormap(){
	FILE* color_fid = fopen("jetcolor", "rb");
	fread(colormap, sizeof(unsigned char), COLORNUM * 3, color_fid);
	fclose(color_fid);
}