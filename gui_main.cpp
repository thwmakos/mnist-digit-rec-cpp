//
// ~thwmakos~
//
// Sat 29 Jun 21:12:05 BST 2024
//
// gui_main.cpp
//
//
// main() function for the GUI build of mnist-digit-rec-cpp
// uses a simple Qt gui to allow drawing
// and then recognising the digit based on the network
// implemented in the other files

#include "qnamespace.h"
#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QPushButton>
#include <QMouseEvent>
#include <QPainter>
#include <QFileDialog>

constexpr auto image_width = 28;
constexpr auto image_height = 28;

class drawing_widget : public QWidget
{
	public:
		drawing_widget(QWidget *parent = nullptr) :
			QWidget(parent), 
			m_drawing(false) 
		{
			setFixedSize(image_width * 20, image_height * 20);
			m_image = QImage(size(), QImage::Format_RGB32);
			m_image.fill(Qt::black);
		}
		
		void save_scaled_image() 
		{
			QString filename = QFileDialog::getSaveFileName(this, "Save Image", "", "PNG (*.png)");
			
			if (!filename.isEmpty()) 
			{
				QImage scaled_image = m_image.scaled(image_width, image_height, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
				QImage grayscale_image = scaled_image.convertToFormat(QImage::Format_Grayscale8);
				grayscale_image.save(filename, "png");
			}
		}

		void clear()
		{
			m_image.fill(Qt::black);
			repaint();
		}

	protected:
		void paintEvent(QPaintEvent *) override 
		{
			QPainter painter(this);
			painter.drawImage(rect(), m_image);
		}

		void mousePressEvent(QMouseEvent *event) override 
		{
			if(event->button() == Qt::LeftButton) 
			{
				m_drawing = true;
				m_last_point = event->pos();
			}
		}

		void mouseMoveEvent(QMouseEvent *event) override 
		{
			constexpr auto pen_size = 25;

			if((event->buttons() & Qt::LeftButton) && m_drawing) 
			{
				QPainter painter(&m_image);
				painter.setPen(QPen(Qt::white, pen_size, Qt::SolidLine, Qt::RoundCap));
				painter.drawLine(m_last_point, event->pos());
				m_last_point = event->pos();
				update();
			}
		}

		void mouseReleaseEvent(QMouseEvent *event) override 
		{
			if (event->button() == Qt::LeftButton) 
			{
				m_drawing = false;
			}
		}
	
	private:
		QImage m_image;
		QPoint m_last_point;
		bool   m_drawing;
	};

class main_window : public QMainWindow
{
	public:
		main_window(QWidget *parent = nullptr) : QMainWindow(parent)
		{
			auto central_widget = new QWidget(this);
			auto layout         = new QVBoxLayout(central_widget);

			m_drawing_widget = new drawing_widget(this);
			layout->addWidget(m_drawing_widget);

			auto clear_button = new QPushButton("Clear", this);
			connect(clear_button, &QPushButton::clicked, m_drawing_widget, &drawing_widget::clear);
			layout->addWidget(clear_button);

			auto save_button = new QPushButton("Save as 28x28 Grayscale", this);
			connect(save_button, &QPushButton::clicked, m_drawing_widget, &drawing_widget::save_scaled_image);
			layout->addWidget(save_button);

			setCentralWidget(central_widget);
			setGeometry(0, 0, image_width * 20, image_height * 20);
			setWindowTitle("mnist-digit-rec-cpp gui");
		}

	private:
		drawing_widget *m_drawing_widget;
};

int main(int argc, char *argv[]) 
{
    QApplication app(argc, argv);
    main_window window;
    window.show();
    return app.exec();
}

