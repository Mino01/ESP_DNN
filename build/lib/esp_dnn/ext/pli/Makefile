CC = clang
CFLAGS = -O3 -Wall -arch arm64 -mmacosx-version-min=11.0
LDFLAGS = -arch arm64


all: bin/pli

bin/pli: src/pli.o
	$(CC) $(LDFLAGS) -o bin/pli src/pli.o

src/pli.o: src/pli.c
	$(CC) $(CFLAGS) -c src/pli.c -o src/pli.o

clean:
	rm -f src/*.o bin/pli

