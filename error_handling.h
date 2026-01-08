#pragma once // Pour ne pas l'inclure 2 fois
#include <hip/hip_runtime.h>
#include <cstdio> // plus léger que <iostream>
#include <cstdlib> // Pour exit() 

// La Fonction Helper (Le Muscle)
// Elle fait le sale boulot : traduire le code d'erreur en texte lisible.
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
   if (code != hipSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// La Macro (Le Sucre Syntaxique)
// Elle capture automatiquement __FILE__ et __LINE__ là où vous l'écrivez.
#define CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }