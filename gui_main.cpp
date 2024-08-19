//
// ~thwmakos~
//
// Sat 29 Jun 21:12:05 BST 2024
//
// gui_main.cpp
//
//
// main() function for the GUI build of mnist-digit-rec-cpp
// uses a simple Qt QML interface to allow drawing
// and then recognising the digit based on the network
// implemented in the other files

#include <QGuiApplication>
#include <QQmlApplicationEngine>

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);
	QQmlApplicationEngine engine("../main_window.qml");

    return app.exec();
}

