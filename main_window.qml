import QtQuick 6.0
import QtQuick.Window 6.0
import QtQuick.Controls 6.0

Window {
	visible: true
	width: 600
	height: 400

	Canvas {
		id: canvas
		anchors.fill: parent
		antialiasing: true

		property var paths: []
		property var currentPath: []

		function drawPath(ctx, path) {
			if(path.length < 2) {
				return;
			}

			ctx.beginPath();
			ctx.moveTo(path[0].x, path[0].y);

			for(var i = 1; i < path.length; i++) {
				ctx.lineTo(path[i].x, path[i].y);
			}

			ctx.stroke();
		}

		onPaint: {
			var ctx = getContext("2d");

			// fill background with black
			ctx.fillStyle = "black";
			ctx.fillRect(0, 0, width, height);

			// drawing color is white
			ctx.lineWidth = 3;
			ctx.strokeStyle = "white";

			for(var i = 0; i < paths.length; i++) {
				drawPath(ctx, paths[i]);
			}

			drawPath(ctx, currentPath);
		}

		MouseArea {
			anchors.fill: parent
			hoverEnabled: true

			onPressed: {
				canvas.currentPath = [{x: mouseX, y: mouseY}];
				canvas.requestPaint();
			}

			onPositionChanged: {
				if(pressed) {
					canvas.currentPath.push({x: mouseX, y: mouseY});
					canvas.requestPaint();
				}
			}

			onReleased: {
				canvas.paths.push(canvas.currentPath);
				canvas.currentPath = [];
				canvas.requestPaint();
			}
		}

		Row {
			spacing: 5

			Button {
				id: button_clear
				text: "Clear"
				onClicked: {
					canvas.paths = [];
					canvas.currentPath = [];
					canvas.requestPaint();
				}
			}

			Button {
				id: button_save
				text: "Save"
				onClicked: {
					canvas.save("image.png");
				}
			}

			Button {
				id: button_quit
				text: "Quit"
				onClicked: {
					Qt.quit();
				}
			}

		}
	}
}
