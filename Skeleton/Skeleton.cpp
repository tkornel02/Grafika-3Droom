//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Toth Kornel
// Neptun : I4XCK0
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const float epsilon = 0.001f;

struct Hit {
	float t;
	vec3 position, normal;
	
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {

public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Light {
	vec3 position;
	vec3 Le;
	Light() {
		position = vec3(0, 0, 0);
		Le = (0, 1, 0);
	}
	Light(vec3 _p, vec3 _Le) {
		position = _p;
		Le = _Le;
	}
};
struct Triangle : Intersectable {
	vec3 a;
	vec3 b;
	vec3 c;

	Triangle(vec3 _r1, vec3 _r2, vec3 _r3) {
		a = _r1;
		b = _r2;
		c = _r3;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 n = (cross(b - a, c - a));
		if (fabs(dot(ray.dir, n)) < epsilon) return hit;
		//float t = (dot(b,n) - dot(ray.start,n)) / (dot(ray.dir, n));
		float t = dot((a - ray.start), n) / dot(ray.dir, n);
		if (t < 0) return hit;
		vec3 p = ray.start + ray.dir * t;
		
		if (intersectbool(p, n)) {
			hit.t = t;
			hit.position = p;
			hit.normal = normalize(n);
		}
		return hit;
	 }

	bool intersectbool(const vec3& p, const vec3& n) {
		float c1, c2, c3;
		c1 = dot(cross(b - a, p - a), n);
		c2 = dot(cross(c - b, p - b), n);
		c3 = dot(cross(a - c, p - c), n);
		if (c1 > 0 && c2 > 0 && c3 > 0) {
			return true;
		}
		return false;
	}
};


struct Cube : public Intersectable {
	std::vector<vec3> vertices;
	std::vector<Triangle> triangles;

	Cube(std::vector<vec3> vs) {
		vertices = vs;
		reloadTriangles();
		
	}

	void reloadTriangles() {
		triangles = {
			Triangle(vertices[0],vertices[6],vertices[4]),
			Triangle(vertices[0],vertices[2],vertices[6]),
			Triangle(vertices[0],vertices[3],vertices[2]),
			Triangle(vertices[0],vertices[1],vertices[3]),
			Triangle(vertices[2],vertices[7],vertices[6]),
			Triangle(vertices[2],vertices[3],vertices[7]),
			Triangle(vertices[4],vertices[6],vertices[7]),
			Triangle(vertices[4],vertices[7],vertices[5]),
			Triangle(vertices[0],vertices[4],vertices[5]),
			Triangle(vertices[0],vertices[5],vertices[1]),
			Triangle(vertices[1],vertices[5],vertices[7]),
			Triangle(vertices[1],vertices[7],vertices[3])

		};
	}


	Hit intersect(const Ray& ray) {
		Hit hit;
		Hit bestHit;
		std::vector<Hit> hits = std::vector<Hit>();

		for (int i = 0; i < 12; i++) {
			hit = triangles[i].intersect(ray);
			if (hit.t > 0)
				hits.push_back(hit);
		}
		if (hits.size() > 0) {
			bestHit = hits[0];
			for (Hit h : hits) {
				if (h.t < bestHit.t)	bestHit = h;
			}
		}
		return bestHit;
	}
	void scale(float s) {
		for (int i = 0; i < vertices.size(); i++) {
			vec4 newV = vec4(vertices[i].x, vertices[i].y, vertices[i].z, 1) * ScaleMatrix(vec3(s, s, s));
			vertices[i] = vec3(newV.x, newV.y, newV.z);
		}
		reloadTriangles();
	}
	void translate(vec3 v) {
		for (int i = 0; i < vertices.size(); i++) {
			vec4 newV = vec4(vertices[i].x, vertices[i].y, vertices[i].z, 1) * TranslateMatrix(v);
			vertices[i] = vec3(newV.x, newV.y, newV.z);
		}
		reloadTriangles();
	}

};
struct Box : public Cube {

	Box(std::vector<vec3> vs) : Cube(vs) {}

	Hit intersect(const Ray& ray) {
		Hit hit;
		Hit bestHit;
		for (int i = 0; i < 12; i++) {
			hit = triangles[i].intersect(ray);
			if (hit.t > bestHit.t) {
				bestHit = hit;
			}
		}
		return bestHit;
	}
	void scale(float s) {
		for (int i = 0; i < vertices.size(); i++) {
			vec4 newV = vec4(vertices[i].x, vertices[i].y, vertices[i].z, 1) * ScaleMatrix(vec3(s, s, s));
			vertices[i] = vec3(newV.x, newV.y, newV.z);
		}
		reloadTriangles();
	}
	void translate(vec3 v) {
		for (int i = 0; i < vertices.size(); i++) {
			vec4 newV = vec4(vertices[i].x, vertices[i].y, vertices[i].z, 1) * TranslateMatrix(v);
			vertices[i] = vec3(newV.x, newV.y, newV.z);
		}
		reloadTriangles();
	}

};

//Okta 
struct Oktaeder : Intersectable {
	std::vector<vec3> vertices;
	std::vector<Triangle> triangles;

	Oktaeder(std::vector<vec3> vs) {
		vertices = vs;
		reloadTriangles();

	}

	void reloadTriangles() {
		triangles = {
			Triangle(vertices[1], vertices[0], vertices[4]),
			Triangle(vertices[2], vertices[1], vertices[4]),
			Triangle(vertices[2], vertices[3], vertices[4]),
			Triangle(vertices[0], vertices[3], vertices[4]),
			Triangle(vertices[0], vertices[1], vertices[5]),
			Triangle(vertices[1], vertices[2], vertices[5]),
			Triangle(vertices[2], vertices[3], vertices[5]),
			Triangle(vertices[3], vertices[0], vertices[5])
		};
	}


	Hit intersect(const Ray& ray) {
		Hit hit;
		Hit bestHit;
		std::vector<Hit> hits = std::vector<Hit>();

		for (int i = 0; i < triangles.size(); i++) {
			hit = triangles[i].intersect(ray);
			if (hit.t > 0)
				hits.push_back(hit);
		}
		if (hits.size() > 0) {
			bestHit = hits[0];
			for (Hit h : hits) {
				if (h.t < bestHit.t)	bestHit = h;
			}
		}
		return bestHit;
	}
	void scale(float s) {
		for (int i = 0; i < vertices.size(); i++) {
			vec4 newV = vec4(vertices[i].x, vertices[i].y, vertices[i].z, 1) * ScaleMatrix(vec3(s, s, s));
			vertices[i] = vec3(newV.x, newV.y, newV.z);
		}
		reloadTriangles();
	}
	void translate(vec3 t) {
		for (int i = 0; i < vertices.size(); i++) {
			vec4 newV = vec4(vertices[i].x, vertices[i].y, vertices[i].z, 1) * TranslateMatrix(vec3(t.x, t.y, t.z));
			vertices[i] = vec3(newV.x, newV.y, newV.z);
		}
		reloadTriangles();
	}
	void rotate(vec3 o) {
		for (int i = 0; i < vertices.size(); i++) {
			vec4 newV = vec4(vertices[i].x, vertices[i].y, vertices[i].z, 1) * RotationMatrix(M_PI / 2, o);
			vertices[i] = vec3(newV.x, newV.y, newV.z);
		}
		reloadTriangles();
	}

};
//Kup
struct Cone : Intersectable {
	float h, al;
	vec3 n, p;
	Light light;
	Cone(float _h, float _al, vec3 _n, vec3 _p, Light _l) {
		h = _h;
		al = _al * M_PI/180;
		n = normalize(_n);
		p = _p;
		light = _l;
	}

	Hit intersect(const Ray& ray)
	{
		Hit hit;

		vec3 sp = ray.start - p;
		float a = dot(ray.dir, n) * dot(ray.dir, n) - cosf(al) * cosf(al);
		float b = 2 * (dot(ray.dir, n) * dot(sp, n) - dot(ray.dir, sp) * cosf(al) * cosf(al));
		float c = dot(sp, n) * dot(sp, n) - dot(sp, sp) * cosf(al) * cosf(al);

		float det = b * b - 4 * a * c;
		if (det <= 0) return hit;

		det = sqrtf(det);
		float t1 = (-b - det) / (2 * a);
		float t2 = (-b + det) / (2 * a);
		float tmin, tmax;

		if (t1 < 0 && t2 < 0) return hit;
		if (t1 > 0 && t2 < 0)  tmin = t1; 
		if (t1 < 0 && t2 > 0) tmin = t2; 
		if (t1 > 0 && t2 > 0) { tmin = t1 < t2 ? t1 : t2; tmax = t1 > t2 ? t1 : t2; }


		/*vec3 p1 = ray.start + ray.dir * t1;
		vec3 p2 = ray.start + ray.dir * t2;*/
		vec3 pmin = ray.start + ray.dir * tmin;
		
		//vec3 rp1 = (ray.start + t1 * ray.dir - p);
		//vec3 rp2 = (ray.start + t2 * ray.dir - p);
		vec3 rpmin = (ray.start + tmin * ray.dir - p);
		vec3 rpmax = (ray.start + tmax * ray.dir - p);
		//vec3 n1 = 2 * dot(rp1, n) * n - 2 * rp1 * cosf(al) * cosf(al);
		//vec3 n2 = 2 * dot(rp2, n) * n - 2 * rp2 * cosf(al) * cosf(al);
		vec3 nmin = 2 * dot(rpmin, n) * n - 2 * rpmin * cosf(al) * cosf(al);
		vec3 nmax = 2 * dot(rpmax, n) * n - 2 * rpmax * cosf(al) * cosf(al);
		//float h1 = dot(rp1, n);
		//float h2 = dot(rp2, n);
		float hmin = dot(rpmin, n);
		float hmax = dot(rpmax, n);

		if (hmin > 0 && hmin < h) {
			hit.t = tmin;
			hit.position = pmin;
			hit.normal = normalize(nmin);
		}
		else {
			if (hmax > 0 && hmax < h)
			hit.t = tmax;
			hit.position = ray.start + ray.dir * tmax;
			hit.normal = normalize(2 * dot((ray.start + tmax * ray.dir - p), n) * n - 2 * (ray.start + tmax * ray.dir - p) * cosf(al) * cosf(al));
		}
		return hit;

		/*if ((h1 < 0 || h1 > h) && (h2 < 0 || h2 > h)) return hit; //nohit
		else if ((h1 < 0 || h1 > h) && (h2 > 0 && h2 < h)) { //t2 valid hit
			
			hit.t = t2;
			hit.position = p2;
			hit.normal = normalize(n2);
		}
		else if ((h1 > 0 && h1 < h) && (h2 < 0 || h2 > h)) { //t1 valid hit
			
			hit.t = t1;
			hit.position = p1;
			hit.normal = normalize(n1);
		}
		else if ((h1 > 0 && h1 < h) && (h2 > 0 && h2 < h)) { //both valid hits : closer
			
			hit.t = tmin;
			hit.position = pmin;
			hit.normal = normalize(nmin);
		}		
		return hit;*/
	}
	void updateCone(vec3 newP, vec3 newN) {
		this->p = newP;
		this->n = normalize(newN);
		light.position = newP +n* 0.08f;
	}
	void setLight(Light l) {
		Light newLight = Light(p + n * 0.08f, l.Le);
		this->light = newLight;
	}
};


class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {

		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}

	vec3 getLookat() { return lookat; }
	vec3 getUp() { return up; }
	vec3 getRight() { return right; }
};



float rnd() { return (float)rand() / RAND_MAX; }

Box* box; Cube* cube; Oktaeder* okta;
Cone* kup1, *kup2,* kup3;
class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Cone*> cones;
	vec3 La; vec3 cEye = vec3(-5.1f, 1.0f, -5.8f); vec3 cUp = vec3(0, 1, 0); vec3 cLookat = vec3(0,0,0);
	float cFov = 45;



public:
	Camera camera;

	void cameraTranslate(vec3 v) {
		vec4 k = vec4(cEye.x, cEye.y, cEye.z, 1);
		vec4 r = k * TranslateMatrix(v) ;
		cEye = vec3(r.x, r.y, r.z);
		//printf("%f", cEye.y);
	}
	void cameraRotate(vec3 v) {
		vec4 k = vec4(cEye.x, cEye.y, cEye.z, 1);
		vec4 r = k * RotationMatrix(M_PI/6,v);
		cEye = vec3(r.x, r.y, r.z);
		//printf("%f", cEye.y);
	}

	void cameraSet(vec3 eye, vec3 vup, vec3 lookat,float fov) {
		cFov = fov * M_PI/180;
		cEye = eye;
		cUp = vup; 
		cLookat = lookat;
	}



	void build() {

		camera.set(cEye, cLookat, cUp, cFov);
		// box csúcspontjai
		std::vector <vec3> vsCube = {
			vec3(0.0f, 0.0f, 0.0f),
			vec3(0.0f, 0.0f, 1.0f),
			vec3(0.0f, 1.0f, 0.0f),
			vec3(0.0f, 1.0f, 1.0f),
			vec3(1.0f, 0.0f, 0.0f),
			vec3(1.0f, 0.0f, 1.0f),
			vec3(1.0f, 1.0f, 0.0f),
			vec3(1.0f, 1.0f, 1.0f)	
		};
	
		std::vector<vec3> vsOkta = {
				vec3(1.0f, 0.0f, 0.0f),
				vec3(0.0f, -1.0f, 0.0f),
				vec3(-1.0f, 0.0f, 0.0f),
				vec3(0.0f, 1.0f, 0.0f),
				vec3(0.0f, 0.0f, 1.0f),
				vec3(0.0f, 0.0f, -1.0f)
		};


		//doboz
		box = new Box(vsCube);
		box->scale(4);
		box->translate(vec3(-2, -2, -2));

		//kocka
		cube = new Cube(vsCube);
		cube->translate(vec3(0.4f, -2.0f, -1));
		
		//Okta
		okta = new Oktaeder(vsOkta);
		okta->translate(vec3(-1, -0.5, 0));
		okta->scale(1.2);


		Light l1; 
		Light l2;
		Light l3;
		//Kupok
		kup1 = new Cone(0.5f, 20, vec3(0, 1, 0), vec3(0, -2, 0), l1);
		kup2 = new Cone(0.5f, 20, vec3(0, 0, -1), vec3(-0.4f, 1.0f, 2.0f), l2);
		kup3 = new Cone(0.5f, 20, vec3(-1, 0, 0), vec3(2.0f, 1.25f, -1.47f), l3);

		kup1->setLight(Light(kup1->p, vec3(1, 0, 0)));
		kup2->setLight(Light(kup2->p, vec3(0, 1, 0)));
		kup3->setLight(Light(kup3->p, vec3(0, 0, 1)));
		cones.push_back(kup1);
		cones.push_back(kup2);
		cones.push_back(kup3);
		
		//hozzaad
		objects.push_back(okta);
		objects.push_back(cube);
		objects.push_back(kup1);
		objects.push_back(kup2);
		objects.push_back(kup3);
		objects.push_back(box);
	}

	void rebuild() {
		camera.set(cEye, cLookat, cUp, cFov);

		objects.clear();
		objects.push_back(box);
		objects.push_back(kup1);
		objects.push_back(kup2);
		objects.push_back(kup3);
		objects.push_back(okta);
		objects.push_back(cube);
		
		
	}
	Cone* closestCone(vec3 p1) {
		float d = length(cones[0]->p - p1);
		Cone* di = cones[0];
		for (Cone* i : cones) {
			float l = length(i->p - p1);
			if (l<d) {
				d = l;
				di = i;
			}
		}
		return di;
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}
	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}
	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t <= 0) return vec3();
		vec3 outRadiance = 0.2 * (1 + dot(hit.normal, -ray.dir)) * vec3(1, 1, 1);
		for (Cone* c : cones) {
			Ray r = Ray(hit.position + hit.t*(epsilon), normalize(c->light.position - hit.position));
			Hit hit2 = firstIntersect(r);
			float angle = dot(hit.normal, r.dir);
			if ( (length(hit2.position - hit.position) > length(c->light.position - hit.position) || hit2.t < 0) && angle > 0) {
				outRadiance = outRadiance + (c->light.Le) /  (powf(length(c->light.position - hit.position),2)+1 );
			}
		}
		return outRadiance;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_DYNAMIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;
std::vector<vec4> image(windowWidth * windowHeight);
// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	scene.build();

	
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {

	if (key == 'a') {
		scene.cameraRotate(vec3(1, 0, 0));
		scene.rebuild();
		scene.render(image);
		fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
		glutPostRedisplay();
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {

	if (button == GLUT_LEFT && state == GLUT_DOWN) {
	Ray r = scene.camera.getRay(pX, windowHeight - pY);
	Hit hitP = scene.firstIntersect(r);
	vec3 p = hitP.position;
	printf("\npos: vec3(%f, %f, %f)", p.x, p.y, p.z);
	printf("\nn: %f, %f, %f", hitP.normal.x, hitP.normal.y, hitP.normal.z);
	scene.closestCone(p)->updateCone(hitP.position, hitP.normal);
	scene.rebuild();
	scene.render(image);
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	glutPostRedisplay();
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {

}