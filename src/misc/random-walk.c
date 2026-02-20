#include<SDL2/SDL.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define WIDTH 900
#define HEIGHT 600
#define AGENT_SIZE 2

int scale = 10;

typedef struct rgb {
  float r, g, b;
} RGB;

typedef struct {
    int vx, vy;
} Velocity;

typedef struct {
    int x,y;
    RGB color;
} Agent;

Velocity get_rand_v()
{
    int choice = rand() / (RAND_MAX/4);
    switch(choice)
    {
        case 0: //up
            return (Velocity) {0,-1};
        case 1: //down
            return (Velocity) {0,1};
        case 2: //left
            return (Velocity) {-1,0};
        case 3: //right
            return (Velocity) {1,0};
    } 
    fprintf(stderr, "Impossible random value %d encountered\n", choice);
    exit(-1);
}

float hue2rgb(float p, float q, float t) {

  if (t < 0) 
    t += 1;
  if (t > 1) 
    t -= 1;
  if (t < 1./6) 
    return p + (q - p) * 6 * t;
  if (t < 1./2) 
    return q;
  if (t < 2./3)   
    return p + (q - p) * (2./3 - t) * 6;
    
  return p;
  
}

RGB hsl2rgb(float h, float s, float l) {

  RGB result;
  
  if(0 == s) {
    result.r = result.g = result.b = l * 255; // achromatic
  }
  else {
    float q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    float p = 2 * l - q;
    result.r = hue2rgb(p, q, h + 1./3) * 255;
    result.g = hue2rgb(p, q, h) * 255;
    result.b = hue2rgb(p, q, h - 1./3) * 255;
  }

  return result;

}

void create_agents(Agent *pagents, int num_agents)
{
    for(int i=0; i<num_agents; i++)
    {
        float h = ((float) rand() / (float) RAND_MAX);
        RGB rgb = hsl2rgb(h, 1, 0.5);
        pagents[i] = (Agent){WIDTH/2, HEIGHT/2, rgb};
    }
}

void move_agent(SDL_Surface *psurface, Agent *pagent)
{
    Velocity v = get_rand_v();
    RGB rgb = pagent->color;
    for(int i=0; i<scale; i++)
    {
        pagent->x += v.vx;
        pagent->y += v.vy;
        SDL_Rect rect = (SDL_Rect) {pagent->x,pagent->y, AGENT_SIZE, AGENT_SIZE};

        // Get SDL color:
        Uint32 color = SDL_MapRGB(psurface->format, rgb.r, rgb.g, rgb.b);
        SDL_FillRect(psurface, &rect, color);
    }
}

int main(int argc, const char *argv[])
{
    int num_agents;
    if(argc == 1)
    {
        num_agents = 5;
    }
    else if(argc == 2)
    {
        num_agents = atoi(argv[1]);
    }
    else
    {
        printf("Usage: %s <num-of-agents>\n", argv[0]);
        return -1;
    }

    srand(time(NULL));

    SDL_Window *pwindow = SDL_CreateWindow("Random Walk", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, 0);
    SDL_Surface *psurface = SDL_GetWindowSurface(pwindow);
    
    Agent *pagents = calloc(num_agents, sizeof(Agent));

    create_agents(pagents, num_agents);

    int app_running = 1;
    while(app_running)
    {
        SDL_Event event;
        while(SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
            {
                app_running = 0;
            }
        }

        for(int i=0; i<num_agents; i++)
        {
            move_agent(psurface, &pagents[i]);
        }
        SDL_UpdateWindowSurface(pwindow);
        SDL_Delay(20);
    }
    free(pagents);
}
