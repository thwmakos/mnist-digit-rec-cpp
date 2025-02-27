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

#include <QApplication>
#include <QMainWindow>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QPushButton>
#include <QLabel>
#include <QProgressBar>
#include <QMouseEvent>
#include <QPainter>
#include <QFileDialog>
#include <QThread>

#include <algorithm>
#include <vector>
#include <array>
#include <print>
#include <memory>

#include "matrix.hpp"
#include "network.hpp"

// in pixels
constexpr auto image_width = 28;
constexpr auto image_height = 28;

constexpr std::array<int, 3> layers = { 28 * 28, 100, 10 };

thwmakos::column_vector softmax(thwmakos::column_vector input)
{
	// undo last layer sigmoid
	std::for_each(input.begin(), input.end(), [](thwmakos::FloatType &item) { item = -std::log(1.0f / item - 1.0f); });
	std::for_each(input.begin(), input.end(), [](thwmakos::FloatType &item) { item = std::exp(item); });

	auto denominator = std::reduce(input.cbegin(), input.cend(), thwmakos::FloatType { 0.0 });

	input *= (1.0f / denominator); 

	return input;
}

class drawing_widget : public QWidget
{
	public:
		drawing_widget(QWidget *parent = nullptr) :
			QWidget(parent), 
			m_drawing(false) 
		{
			setFixedSize(image_width * 12, image_height * 12);
			m_image = QImage(size(), QImage::Format_Grayscale8);
			m_image.fill(Qt::black);
		}
		
		void clear()
		{
			m_image.fill(Qt::black);
			repaint();
		}
		
		// return the drawn shape as a 28 * 28 matrix suitable for
		// evaluation by the network
		thwmakos::column_vector get_drawing()
		{
			std::vector<thwmakos::FloatType> pixels(28 * 28);
			
			auto scaled_image = m_image.scaled(image_width, image_height, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
			auto data = scaled_image.constBits();

			// expect 28 * 28 pixel, 1 byte in size each
			if(scaled_image.sizeInBytes() != image_width * image_height)
			{
				std::println("image size in bytes {}, expected {}", scaled_image.sizeInBytes(), 28 * 28);	
			}

			std::transform(data, data + 28 * 28, pixels.begin(), [] (uchar pix) { return static_cast<thwmakos::FloatType>(pix) / 255.0f; });

			return thwmakos::column_vector(28 * 28, std::move(pixels));
		}

		void save_scaled_image()
		{
			QString filename = QFileDialog::getSaveFileName(this, "Save Image", "", "PNG (*.png)");
			
			if(!filename.isEmpty()) 
			{
				QImage scaled_image = m_image.scaled(image_width, image_height, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
				scaled_image.save(filename, "png");
			}
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
		main_window(QWidget *parent = nullptr) : QMainWindow(parent), nwk(layers)
		{
			auto central_widget = new QWidget(this);
		
			// main layout, consisting of a left and a right layout
			// the left shows the drawing area and buttons
			// the right displays the softmax evaluation with sliders
			auto grid_layout    = new QGridLayout(central_widget);

			auto layout         = new QVBoxLayout();

			m_drawing_widget = new drawing_widget(this);
			layout->addWidget(m_drawing_widget);

			m_eval_label = new QLabel("Draw a number");
			m_eval_label->setAlignment(Qt::AlignCenter);
			layout->addWidget(m_eval_label);

			m_raw_eval_label = new QLabel("Raw network evaluation"); 
			m_raw_eval_label->setAlignment(Qt::AlignCenter);
			m_raw_eval_label->setWordWrap(true);
			layout->addWidget(m_raw_eval_label);

			m_eval_button = new QPushButton("Evaluate drawing");
			connect(m_eval_button, &QPushButton::clicked, this, &main_window::evaluate_drawing);
			layout->addWidget(m_eval_button);

			auto clear_button = new QPushButton("Clear");
			connect(clear_button, &QPushButton::clicked, m_drawing_widget, &drawing_widget::clear);
			layout->addWidget(clear_button);

			m_train_button = new QPushButton("Train network");
			connect(m_train_button, &QPushButton::clicked, this, &main_window::dispatch_training); 
			layout->addWidget(m_train_button);

			auto save_button = new QPushButton("Save as 28x28 grayscale");
			connect(save_button, &QPushButton::clicked, m_drawing_widget, &drawing_widget::save_scaled_image);
			layout->addWidget(save_button);

			auto quit_button = new QPushButton("Quit");
			connect(quit_button, &QPushButton::clicked, qApp, &QApplication::quit);
			layout->addWidget(quit_button);

			// softmax display column on the right of window
			
			// put everything in a groupbox
			auto groupbox = new QGroupBox("Softmax evaluation");

			auto softmax_layout = new QFormLayout();

			for(int i = 0; i < 10; ++i)
			{
				m_bars[i] = new QProgressBar();
				m_bars[i]->setRange(0, 100);
				m_bars[i]->setValue(0);
				softmax_layout->addRow(QString::fromStdString(std::format("{}:", i)), m_bars[i]);
			}
			
			//groupbox->setFlat(true);
			groupbox->setLayout(softmax_layout);	
			grid_layout->addLayout(layout, 0, 0);
			grid_layout->addWidget(groupbox, 0, 1);
			grid_layout->setSizeConstraint(QLayout::SetFixedSize);

			setCentralWidget(central_widget);
			//setGeometry(100, 100, image_width * 20, image_height * 20);
			setWindowTitle("mnist-digit-rec-cpp gui");
			//setFixedSize(sizeHint());
			//setWindowIcon(QIcon(QString("../data/icon.png")));
		}

		void evaluate_drawing()
		{
			auto drawing = m_drawing_widget->get_drawing();
			auto eval    = nwk.evaluate(drawing);
			
			auto text = std::format("The number you drew is <b>{}</b>", thwmakos::output_to_int(eval));

			m_eval_label->setText(QString::fromStdString(text));

			std::string raw_eval_text {};
			
			for(int i = 0; i < eval.length(); ++i)
			{
				raw_eval_text += std::format("<b>{}</b>: {:.2} ", i, eval[i]); 
			}

			m_raw_eval_label->setText(QString::fromStdString(raw_eval_text));

			auto softmax_eval = softmax(std::move(eval));

			for(int i = 0; i < softmax_eval.length(); ++i)
			{
				m_bars[i]->setValue(softmax_eval[i] * 100);
			}
		}

		void dispatch_training()
		{
			m_eval_label->setText("Network training in progress");
			m_eval_button->setDisabled(true);
			m_train_button->setDisabled(true);

			m_training_thread.reset(QThread::create([this] { nwk.train(30, 100, 3.0f, 10.0f); }));
			
			connect(m_training_thread.get(), &QThread::finished, 
					[this] 
					{ 
						m_eval_label->setText("Network training complete"); 
						m_eval_button->setEnabled(true);
						m_train_button->setEnabled(true);
					});

			m_training_thread->start();
		}

	private:
		drawing_widget *m_drawing_widget = nullptr;
		QLabel         *m_eval_label     = nullptr;
		QLabel         *m_raw_eval_label = nullptr;
		QPushButton    *m_eval_button    = nullptr;
		QPushButton    *m_train_button   = nullptr;
		
		// one progress bar for each digit that shows
		// the probability the network this the drawing
		// shows
		std::array<QProgressBar *, 10> m_bars = { nullptr };

		std::unique_ptr<QThread> m_training_thread = nullptr;
		thwmakos::network nwk;
};

int main(int argc, char *argv[]) 
{
    QApplication app(argc, argv);
    main_window window;
    window.show();
    return app.exec();
}

