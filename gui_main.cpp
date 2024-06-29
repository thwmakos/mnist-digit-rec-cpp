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
#include <QQuickView>
#include <QQmlContext>

#include <memory>

int main(int argc, char *argv[])
{
	// all I need is a simple window to load the qml file,
	// QQuickView seems ideal for this
    QGuiApplication app(argc, argv);
    auto view = std::make_unique<QQuickView>();

	// handle closing the window properly	
	QObject::connect(view->engine(), &QQmlApplicationEngine::quit, &app, &QGuiApplication::quit);

    view->setSource(QUrl("file:../main_window.qml"));
   	view->show();

    return app.exec();
}

