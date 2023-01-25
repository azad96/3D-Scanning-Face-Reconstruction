#include <iostream>
#include <string>
#include <vtkSmartPointer.h>
//#include <vtkOBJReader.h>
#include <vtkPLYReader.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleRubberBandPick.h>
#include <vtkProperty.h>
#include <vtkPLYWriter.h>

#include <stdio.h>

#include "optimal_nonrigid_icp.h"

int main(int argc, char**argv)
{
	std::string template_filename = argv[1];
	std::string target_filename = argv[2];

	// vtkSmartPointer<vtkOBJReader> template_reader = vtkSmartPointer<vtkOBJReader>::New();
	// template_reader->SetFileName(template_filename.c_str());
	// template_reader->Update();

	// vtkSmartPointer<vtkOBJReader> target_reader = vtkSmartPointer<vtkOBJReader>::New();
	// target_reader->SetFileName(target_filename.c_str());
	// target_reader->Update();

	vtkSmartPointer<vtkPLYReader> source_reader = vtkSmartPointer<vtkPLYReader>::New();
	source_reader->SetFileName(template_filename.c_str());
	source_reader->Update();

	vtkSmartPointer<vtkPLYReader> template_reader = vtkSmartPointer<vtkPLYReader>::New();
	template_reader->SetFileName(template_filename.c_str());
	template_reader->Update();

	vtkSmartPointer<vtkPLYReader> target_reader = vtkSmartPointer<vtkPLYReader>::New();
	target_reader->SetFileName(target_filename.c_str());
	target_reader->Update();

  	vtkSmartPointer<vtkPolyData> source_polyData = source_reader->GetOutput();
  	vtkSmartPointer<vtkPolyData> template_polyData = template_reader->GetOutput();
	vtkSmartPointer<vtkPolyData> target_polyData = target_reader->GetOutput();

	OptimalNonrigidICP oni(template_polyData, target_polyData);
	oni.init();
	std::cout << "init() finished!" << std::endl;


	vtkSmartPointer<vtkPLYWriter> plyWriter = vtkSmartPointer<vtkPLYWriter>::New();
	plyWriter->SetInputData(template_polyData);

	float alpha = 10.0;
	float beta = 1.0;
	float gamma = 1.0;
	int step = 2;

	for (int i = 0; i < step; ++i)
	{
		oni.initCompute();
		std::cout << "initCompute() finished!" << std::endl;
		oni.compute(alpha, beta, gamma);
		std::cout << "compute() finished!" << std::endl;

		char num_str[100];
		sprintf(num_str, "%d", i);
		plyWriter->SetFileName( (template_filename.substr(0, template_filename.find_last_of('.')) + "_" + num_str).c_str() );
		plyWriter->Write();
	}

	vtkSmartPointer<vtkPolyDataMapper> source_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
 	source_mapper->SetInputData(source_polyData);	

	vtkSmartPointer<vtkPolyDataMapper> template_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
 	template_mapper->SetInputData(template_polyData);	

  	vtkSmartPointer<vtkPolyDataMapper> target_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  	target_mapper->SetInputData(target_polyData);	

	vtkSmartPointer<vtkActor> source_actor = vtkSmartPointer<vtkActor>::New();
  	source_actor->SetMapper(source_mapper);
  	source_actor->GetProperty()->SetColor(1,1,1);

  	vtkSmartPointer<vtkActor> template_actor = vtkSmartPointer<vtkActor>::New();
  	template_actor->SetMapper(template_mapper);
  	template_actor->GetProperty()->SetColor(0.8,0.2,0);

  	vtkSmartPointer<vtkActor> target_actor = vtkSmartPointer<vtkActor>::New();
  	target_actor->SetMapper(target_mapper);
  	target_actor->GetProperty()->SetColor(0,0.2,0.8);

  	vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
  	vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
  	renderWindow->AddRenderer(renderer);
  	vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
  	renderWindowInteractor->SetRenderWindow(renderWindow);

  	renderer->AddActor(source_actor);
  	renderer->AddActor(template_actor);
  	// renderer->AddActor(target_actor);
  	renderer->SetBackground(0.1804,0.5451,0.3412); // Sea green

	vtkSmartPointer<vtkInteractorStyleRubberBandPick> interactorStyle = vtkSmartPointer<vtkInteractorStyleRubberBandPick>::New();
	renderWindowInteractor->SetInteractorStyle(interactorStyle);

  	renderWindow->Render();
  	renderWindowInteractor->Start();

	return 0;
}
