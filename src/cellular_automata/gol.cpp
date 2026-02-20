#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <SDL2/SDL.h>

Uint32 COLOR_WHITE = 0xffffffff;
Uint32 COLOR_GRAY = 0x2f2f2f2f;
Uint32 COLOR_BLACK = 0x00000000;
int SURFACE_WIDTH = 1200;
int SURFACE_HEIGHT = 900;
int CELL_WIDTH = 5;
int LINE_WIDTH = 0;

int draw_cell(SDL_Surface* surface, int cell_x, int cell_y, int cell_value)
{
	int pixel_x = cell_x * CELL_WIDTH;
	int pixel_y = cell_y * CELL_WIDTH;
	Uint32 color = cell_value == 0 ? COLOR_BLACK : COLOR_WHITE;

	SDL_Rect cell_rect = (SDL_Rect) {pixel_x, pixel_y, CELL_WIDTH, CELL_WIDTH};
	SDL_FillRect(surface, &cell_rect, color);
}

int draw_grid(SDL_Surface* surface, int columns, int rows)
{
	for (int i=0; i<rows; i++)
	{
		SDL_Rect line = (SDL_Rect) {0, i*CELL_WIDTH, SURFACE_WIDTH, LINE_WIDTH};
		SDL_FillRect(surface, &line, COLOR_GRAY);
	}
	for (int i=0; i<columns; i++)
	{
		SDL_Rect line = (SDL_Rect) {i*CELL_WIDTH, 0, LINE_WIDTH, SURFACE_HEIGHT};
		SDL_FillRect(surface, &line, COLOR_GRAY);
	}
}

void draw_game_matrix(SDL_Surface* surface, int rows, int columns, int game_matrix[])
{
	for (int i=0; i<rows; i++)
	{
		for (int j=0; j<columns; j++)
		{
			int cell_value = game_matrix[j + columns * i];
			draw_cell(surface, j, i, cell_value);
		}
	}
}

void randomize_game_matrix(int rows, int columns, int game_matrix[])
{
	for (int i=0; i<rows; i++)
	{
		for (int j=0; j<columns; j++)
		{
			game_matrix[j + columns * i] = rand() % 2;
		}
	}
}

void blank_game_matrix(int rows, int columns, int game_matrix[])
{
	for (int i=0; i<rows; i++)
	{
		for (int j=0; j<columns; j++)
		{
			game_matrix[j + columns * i] = 0;
		}
	}
}

int count_neighbors(int i, int j, int row_count, int column_count, int game_matrix[])
{
	if(i >= row_count)
		printf("ERROR: i bigger than row_count\n");
	if(j >= column_count)
		printf("ERROR: j bigger than column_count\n");
		
	int neighbor_counter = 0;
	// lefthand neighbor
	if (j > 0)
	{
//		printf("left neighbor found\n");
		neighbor_counter += game_matrix[j-1 + column_count * i];
	}
	// righthand neighbor
	if (j < (column_count - 1))
	{
//		printf("right neighbor found\n");
		neighbor_counter += game_matrix[j+1 + column_count * i];
	}
	// above neighbor
	if (i > 0)
	{
//		printf("above neighbor found\n");
		neighbor_counter += game_matrix[j + column_count * (i-1)];
	}
	// above left neighbor
	if (i > 0 && j > 0)
	{
//		printf("above left neighbor found\n");
		neighbor_counter += game_matrix[j-1 + column_count * (i-1)];
	}
	// above right neighbor
	if (i > 0 && j < (column_count - 1))
	{
//		printf("above right neighbor found\n");
		neighbor_counter += game_matrix[j+1 + column_count * (i-1)];
	}
	// below neighbor
	if (i < (row_count - 1))
	{
//		printf("below neighbor found\n");
		neighbor_counter += game_matrix[j+ column_count * (i+1)];
	}
	// below left neighbor
	if (i < (row_count - 1) && j > 0)
	{
//		printf("below left neighbor found\n");
		neighbor_counter += game_matrix[j-1 + column_count * (i+1)];
	}
	// below right neighbor
	if (i < (row_count - 1) && j < (column_count - 1))
	{
//		printf("below right neighbor found\n");
		neighbor_counter += game_matrix[j+1 + column_count * (i+1)];
	}
	return neighbor_counter;
}

void simulation_step(int rows, int columns, int game_matrix[])
{
	int next_game_matrix[rows*columns];

	for (int i=0; i<rows; i++)
	{
		for (int j=0; j<columns; j++)
		{
			int neighbor_count = count_neighbors(i,j,rows,columns, game_matrix);
			// perform logic based on neighbor count
			int current_cell_value = game_matrix[j + columns*i];
			
			// rule 1
			if (current_cell_value != 0 && neighbor_count < 2)
			{
				next_game_matrix[j + columns*i] = 0;
			}
			// rule 2
			else if (current_cell_value != 0 && (neighbor_count == 2 || neighbor_count == 3))
			{
				next_game_matrix[j + columns*i] = 1;
			}
			// rule 3
			else if (current_cell_value != 0 && neighbor_count > 3)
			{
				next_game_matrix[j + columns*i] = 0;
			}
			// rule 4
			else if (current_cell_value == 0 && neighbor_count == 3)
			{
				next_game_matrix[j + columns*i] = 1;
			}
			else
			{
				next_game_matrix[j + columns*i] = current_cell_value;
			}
		}
	}

	for (int i=0; i<rows*columns; i++)
		game_matrix[i] = next_game_matrix[i];
}

void set_cell_in_game_matrix(Sint32 mouse_x, Sint32 mouse_y, int rows, int columns, int game_matrix[])
{
	int j = mouse_x / CELL_WIDTH;
	int i = mouse_y / CELL_WIDTH;
	game_matrix[j + columns*i] = !game_matrix[j + columns*i];
}


int main()
{
	printf("Hello Conway's Game of Life\n");
	// Seeding the random number generator
	srand ( time(NULL) );

	SDL_Init(SDL_INIT_VIDEO);

	char* window_title = "Conway's Game of Life";

	int columns = SURFACE_WIDTH / CELL_WIDTH;
	int rows = SURFACE_HEIGHT / CELL_WIDTH;

	SDL_Window* window = SDL_CreateWindow(window_title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, SURFACE_WIDTH, SURFACE_HEIGHT, 0);

	SDL_Surface* surface = SDL_GetWindowSurface(window);

	int row_count = SURFACE_HEIGHT / CELL_WIDTH;
	int column_count = SURFACE_WIDTH / CELL_WIDTH;
	printf("rows=%d columns=%d\n", row_count, column_count);
	int game_matrix[row_count * column_count];	

	int fake_game[] = {1,1,1,1,1,1,1,1,1};
	int fake_count = count_neighbors(2,1,3,3,fake_game);
	printf("Fake count=%d\n", fake_count);
	
	randomize_game_matrix(row_count, column_count, game_matrix);
	draw_game_matrix(surface, row_count, column_count, game_matrix);
	draw_grid(surface, columns, rows);
	SDL_UpdateWindowSurface(window);
	int game_loop = 1;
	int simulation_paused = 1;
	SDL_Event event;
	while (game_loop)
	{
		while(SDL_PollEvent(&event))
		{
			if (event.type == SDL_QUIT)
			{
				game_loop = 0;
			}
			else if (event.type == SDL_KEYDOWN)
			{
				if (event.key.keysym.sym == SDLK_SPACE)
				{
					simulation_paused = !simulation_paused;
				}
				if (event.key.keysym.sym == SDLK_RETURN)
				{
					randomize_game_matrix(row_count, column_count, game_matrix);
					draw_game_matrix(surface, row_count, column_count, game_matrix);
					draw_grid(surface, columns, rows);
					SDL_UpdateWindowSurface(window);
				}
				if (event.key.keysym.sym == SDLK_BACKSPACE)
				{
					blank_game_matrix(row_count, column_count, game_matrix);
					draw_game_matrix(surface, row_count, column_count, game_matrix);
					draw_grid(surface, columns, rows);
					SDL_UpdateWindowSurface(window);
				}
			}
			else if (event.type == SDL_MOUSEBUTTONDOWN)
			{
				Sint32 mouse_x = event.button.x;
				Sint32 mouse_y = event.button.y;
				set_cell_in_game_matrix(mouse_x, mouse_y, row_count, column_count, game_matrix);
				draw_game_matrix(surface, row_count, column_count, game_matrix);
				draw_grid(surface, columns, rows);
				SDL_UpdateWindowSurface(window);
			}
		}
		if ( ! simulation_paused)
		{
			simulation_step(row_count, column_count, game_matrix);
			draw_game_matrix(surface, row_count, column_count, game_matrix);
			draw_grid(surface, columns, rows);
			SDL_UpdateWindowSurface(window);
		}
		SDL_Delay(100);
	}
}
