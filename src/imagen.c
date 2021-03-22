#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include "../include/derivados_mpi.h"
#include "../include/imagen.h"

extern int rank, world_size;
extern MPI_Datatype rgb_type, individuo_type;

RGB *leer_ppm(const char *file, int *ancho, int *alto, int *max)
{
	int i, n;
	FILE *fd;
	
	char c, b[100];
	int red, green, blue;
	
	fd = fopen(file, "r");
	
	n = fscanf(fd, "%[^\n] ", b);
	if(b[0] != 'P' || b[1] != '3') {
		printf("%s no es una imagen PPM\n", file);
		exit(0);
	}
	
	printf("Leyendo fichero PPM %s\n", file);
	n = fscanf(fd, "%c", &c);
	while(c == '#') {
		n = fscanf(fd, "%[^\n] ", b);
		printf("%s\n", b);
		n = fscanf(fd, "%c", &c);
	}
	
	ungetc(c, fd);
	n = fscanf(fd, "%d %d %d", ancho, alto, max);
	assert(n == 3);
	
	int size = (*ancho)*(*alto);
	
	RGB *imagen = (RGB *) malloc(size*sizeof(RGB));
	assert(imagen);
	
	for(i=0; i<size; i++) {
		n = fscanf(fd, "%d %d %d", &red, &green, &blue);
		assert(n == 3);
		
		imagen[i].r = red;
		imagen[i].g = green;
		imagen[i].b = blue;
	}
	
	fclose(fd);
	return imagen;
}

void escribir_ppm(const char *fichero, int ancho, int alto, int max, const RGB *imagen)
{
	int i;
	FILE *fd;
	
	fd = fopen(fichero, "w");
	
	fprintf(fd, "P3\n");
	fprintf(fd, "%d %d\n%d\n", ancho, alto, max);
	
	int size = alto*ancho;
	for(i=0; i<size; i++) {
		fprintf(fd, "%d %d %d ", imagen[i].r, imagen[i].g, imagen[i].b);
		if((i+1) % 18 == 0) { fprintf(fd, "\n"); }
	}
	fclose(fd);
}

void suavizar(int ancho, int alto, RGB *imagen)
{
	// Aplicar tecnica "mean-filter" para suavizar la imagen resultante
	int r = 0, g = 0, b = 0;
	//	(i-1, j-1) | (i-1, j) | (i-1, j+1)
	//	(i, j-1)   | p(i,j)   | (i, j+1)
	//	(i+1, j-1) | (i+1, j) | (i+1, j+1)
  MPI_Status status;
  int chunk_size = alto / world_size;
  int leftover = alto % world_size;
  int offset;
  int position;
  int buff_size = sizeof(int) + sizeof(RGB) * ancho * chunk_size;
  RGB * packBuff = (RGB *) malloc(buff_size);
  if(rank == 0) {
    offset = ancho*(chunk_size + leftover);
    for(int i = 1; i < world_size; i++) {
      position = 0;
      MPI_Pack(&offset, 1, MPI_INT, packBuff, buff_size, &position, MPI_COMM_WORLD); 
      MPI_Pack(&(imagen[offset]), ancho * chunk_size, rgb_type, packBuff, buff_size, &position, MPI_COMM_WORLD);
      MPI_Send(packBuff, position, MPI_PACKED, i, 0, MPI_COMM_WORLD);

      offset += ancho * chunk_size;
    }
    chunk_size += leftover;
  } else {
    imagen = (RGB *) malloc(ancho * chunk_size * sizeof(RGB));
    position = 0;
    MPI_Recv(packBuff, buff_size, MPI_PACKED, MASTER, 0, MPI_COMM_WORLD, &status);
    MPI_Unpack(packBuff, buff_size, &position, &offset, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(packBuff, buff_size, &position, imagen, ancho * chunk_size, rgb_type, MPI_COMM_WORLD);
  }

	for(int i = 1; i < chunk_size - 1; i++) {
		for(int j = 1; j < ancho - 1; j++) {
			r = imagen[(i-1)*ancho + j-1].r +  imagen[(i-1)*ancho + j].r +  imagen[(i-1)*ancho + j+1].r + 
			imagen[i*ancho + j-1].r + imagen[i*ancho + j].r + imagen[i*ancho + j+1].r + 
			imagen[(i+1)*ancho + j-1].r +  imagen[(i+1)*ancho + j].r +  imagen[(i+1)*ancho + j+1].r;

			g = imagen[(i-1)*ancho + j-1].g +  imagen[(i-1)*ancho + j].g +  imagen[(i-1)*ancho + j+1].g +
			imagen[i*ancho + j-1].g + imagen[i*ancho + j].g + imagen[i*ancho + j+1].g +
			imagen[(i+1)*ancho + j-1].g +  imagen[(i+1)*ancho + j].g +  imagen[(i+1)*ancho + j+1].g;

			b = imagen[(i-1)*ancho + j-1].b +  imagen[(i-1)*ancho + j].b +  imagen[(i-1)*ancho + j+1].b +
			imagen[i*ancho + j-1].b + imagen[i*ancho + j].b + imagen[i*ancho + j+1].b +
			imagen[(i+1)*ancho + j-1].b +  imagen[(i+1)*ancho + j].b +  imagen[(i+1)*ancho + j+1].b;

			imagen[i*ancho + j].r = round(r/9);
			imagen[i*ancho + j].g = round(g/9);
			imagen[i*ancho + j].b = round(b/9);
		}
	}
  /* Bordes superior e inferior */
  for (int j = 1; j < ancho - 1; j++) {
    imagen[j] = imagen[j + ancho];
    imagen[ancho*(chunk_size-1)+j] = imagen[ancho*(chunk_size-1)+ j - ancho];
  }

  if(rank == MASTER) {
    chunk_size -= leftover;
    for(int i = 1; i < world_size; i++) {
      position = 0;
      MPI_Recv(packBuff, buff_size, MPI_PACKED, i, 0, MPI_COMM_WORLD, &status);
      MPI_Unpack(packBuff, buff_size, &position, &offset, 1, MPI_INT, MPI_COMM_WORLD);
      MPI_Unpack(packBuff, buff_size, &position, &(imagen[offset]), ancho * chunk_size, rgb_type, MPI_COMM_WORLD);
    }
    // Suavizar bordes
    /* Esquinas */
    imagen[0] = imagen[ancho+1];
    imagen[ancho-1] = imagen[ancho*2 + 2];
    imagen[ancho*(alto-1)] = imagen[ancho*(alto-1) - ancho + 1];
    imagen[ancho*alto - 1] = imagen [ancho*alto - ancho - 2];

    /* Bordes izquierda y derecha */
    for (int i = 1; i < alto - 1; i++) {
      imagen[ancho*i] = imagen[ancho*i + 1];
      imagen[ancho*i - 1] = imagen[ancho*i - 2];
    }

  } else {
      position = 0;
      MPI_Pack(&offset, 1, MPI_INT, packBuff, buff_size, &position, MPI_COMM_WORLD); 
      MPI_Pack(imagen, ancho * chunk_size, rgb_type, packBuff, buff_size, &position, MPI_COMM_WORLD);
      MPI_Send(packBuff, position, MPI_PACKED, MASTER, 0, MPI_COMM_WORLD);
  }
  free(packBuff);
}
