#include <stdio.h>
#include <SDL2/SDL.h>
#include <math.h>


#define WIDTH 900
#define HEIGHT 600
#define COLOR_WHITE 0xffffffff
#define COLOR_BLACK 0x00000000
#define COLOR_GRAY 0xf1f1f1f1
#define COLOR_BACKGROUND 0x1b1b1b1b
#define COLOR_TRAJECTORY 0x4cff42
#define COLOR_TRAJECTORY2 0xff1430
#define DAMPENING 0.98
#define DELTA_T 0.1
#define TRAJECTORY_LENGTH 20000
#define TRAJECTORY_WIDTH 2

struct Circle
{
    double x;
    double y;
    double radius;
    double v_x;
    double v_y;
    double mass;
};

void FillCircle(SDL_Surface* surface, struct Circle circle, Uint32 color)
{
    double low_x = circle.x - circle.radius;
    double low_y = circle.y - circle.radius;
    double high_x = circle.x + circle.radius;
    double high_y = circle.y + circle.radius;
    
    double radius_squared = circle.radius * circle.radius;
        
    for (double x = low_x; x < high_x; x++)
    {
        for (double y = low_y; y < high_y; y++)
        {
            // is coordinate within circle?
            double center_distance_squared = (x-circle.x)*(x-circle.x) + (y-circle.y)*(y-circle.y);     
            if (center_distance_squared < radius_squared / 10 )
            {
                SDL_Rect pixel = (SDL_Rect) {x,y,1,1};
                SDL_FillRect(surface, &pixel, color);

            }
            
        }    
    }
}

void screen_boundary_step(struct Circle* circle)
{
    if ( circle->x + circle->radius > WIDTH)
    {
        circle->x = WIDTH - circle->radius;
        circle->v_x = -circle->v_x * DAMPENING;
    }
    if ( circle->y + circle->radius > HEIGHT)
    {
        circle->y = HEIGHT - circle->radius;
        circle->v_y = -circle->v_y * DAMPENING;
    }
    if ( circle->y - circle->radius < 0)
    {
        circle->y = circle->radius;
        circle->v_y = -circle->v_y * DAMPENING;
    }    
    if ( circle->x - circle->radius < 0)
    {
        circle->x = circle->radius;
        circle->v_x = -circle->v_x * DAMPENING;
    }    
}

void step(struct Circle* circle, struct Circle* circle2)
{
    // how do we calculate the new position?
    circle->x += circle->v_x * DELTA_T;
    circle->y += circle->v_y * DELTA_T;
    circle2->x += circle2->v_x * DELTA_T;
    circle2->y += circle2->v_y * DELTA_T;

    // Update gravitational acceleration
    double distance =fmax(circle->radius + circle2->radius, sqrt( pow( circle->x - circle2->x, 2) + pow(circle->y - circle2->y, 2) ));
    double ndx = (circle2->x - circle->x) / distance;
    double ndy = (circle2->y - circle->y) / distance;
    double a_gravity = 100 / pow(distance,2);
    double ax = a_gravity * ndx;    
    double ay = a_gravity * ndy;

    circle->v_x += ax*circle2->mass;
    circle->v_y += ay*circle2->mass;
    circle2->v_x += -ax*circle->mass;
    circle2->v_y += -ay*circle->mass;

    // did the ball exit the screen?
//    screen_boundary_step(circle);
 //   screen_boundary_step(circle2);
}

void FillTrajectory(SDL_Surface* surface, struct Circle trajectory[TRAJECTORY_LENGTH], Uint32 color)
{
	for (int i=0; i<TRAJECTORY_LENGTH; i++)
	{
		double trajectory_size = TRAJECTORY_WIDTH;
		trajectory[i].radius = trajectory_size;
		FillCircle(surface, trajectory[i], color);
	}

}

void UpdateTrajectory(struct Circle trajectory[TRAJECTORY_LENGTH], struct Circle circle)
{
    // shift array - write the circle at the end of the array
    struct Circle trajectory_shifted_copy[TRAJECTORY_LENGTH];
    for (int i=1; i<TRAJECTORY_LENGTH; i++)
    {
        trajectory_shifted_copy[i-1] = trajectory[i];
    }
    for (int i=0; i<TRAJECTORY_LENGTH; i++)
        trajectory[i] = trajectory_shifted_copy[i];

    trajectory[TRAJECTORY_LENGTH-1] = circle;
}

int main()
{
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("Bouncy Ball", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_BORDERLESS);
    SDL_Surface* surface = SDL_GetWindowSurface(window);

    struct Circle circle = (struct Circle) {200, 200, 25, 13, 0, 200};
    struct Circle circle2 = (struct Circle) {400, 400, 25, -7, 0, 371.428571429};

    SDL_Rect erase_rect = (SDL_Rect){0,0,WIDTH,HEIGHT};    
    SDL_Event event;
    struct Circle trajectory[TRAJECTORY_LENGTH];
    struct Circle trajectory2[TRAJECTORY_LENGTH];
    int simulation_running = 1;
    while(simulation_running)
    {
        while(SDL_PollEvent(&event))
        {
            if(event.type == SDL_QUIT)
            {
                simulation_running = 0;
            }
            if(event.type == SDL_KEYDOWN)
            {
                if(event.key.keysym.sym == SDLK_SPACE)
                {
                    simulation_running = 0;
                }
            }
        }

        SDL_FillRect(surface, &erase_rect, COLOR_BACKGROUND);
        FillTrajectory(surface, trajectory, COLOR_TRAJECTORY);
        FillTrajectory(surface, trajectory2, COLOR_TRAJECTORY2);
        FillCircle(surface, circle, COLOR_WHITE);
        FillCircle(surface, circle2, COLOR_WHITE);
        step(&circle, &circle2);
	UpdateTrajectory(trajectory, circle);
	UpdateTrajectory(trajectory2, circle2);
        SDL_UpdateWindowSurface(window);
        SDL_Delay(10);
    }

}
