CC=g++
CFLAGS= -I /usr/include/eigen3
INCLUDE_DIR = -I./
INCLUDE_DIR+= -I./depends/raylib/raylib/include
INCLUDE_DIR+= -I./depends/raygui/src
LIBRARY_DIR = -L./depends/raylib/raylib
LIBRARY_DIR+= -L./depends/raylib/raylib/external/glfw/src
LIBRARY_DIR+= -L/usr/lib/x86_64-linux-gnu
LIBRARY_LINK = -lraylib -lglfw -ldl -pthread
BUILD_DEP = $(INCLUDE_DIR) $(LIBRARY_DIR) $(LIBRARY_LINK)

all: mmatching

clean:
	-rm mmatching

mmatching: controller.cpp
	$(CC) controller.cpp -o $@ $(BUILD_DEP)

run:
	./mmatching;

build_depends:
	mkdir -p depends/raygui
	mkdir -p depends/raylib
	git clone --depth 1 https://github.com/raysan5/raylib.git depends/raylib
	git clone --depth 1 https://github.com/raysan5/raygui depends/raygui
	cd depends/raylib && cmake . && make

patch-back:
	mv controller.old.cpp controller.cpp

.SILENT: clean

