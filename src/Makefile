CC = gcc
CFLAGS = -O3 -std=c99 -g -Wall

sec: main.c imagen.c ga.c
	$(CC) $(CFLAGS) main.c imagen.c ga.c -o sec -DTIME -DDEBUG -lm

test_sec:
	./sec ../input/pitufo.ppm ../output/pitufo-out.ppm 15000 256

clean:
	rm -f sec
