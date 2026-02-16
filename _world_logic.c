#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

/* RNG state */
static uint64_t state = 88172645463325252ull;

/* xorshift64* */
static inline uint64_t xorshift64star() {
    uint64_t x = state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state = x;
    return x * 2685821657736338717ull;
}

/* Python wrapper: random() -> float in [0,1) */
static PyObject* fast_random(PyObject* self, PyObject* args) {
    uint64_t r = xorshift64star();

    /* Convert to double in [0,1) using top 53 bits */
    double result = (r >> 11) * (1.0 / 9007199254740992.0);

    return PyFloat_FromDouble(result);
}

static double dFastRandom() {
    uint64_t r = xorshift64star();

    /* Convert to double in [0,1) using top 53 bits */
    double result = (r >> 11) * (1.0 / 9007199254740992.0);

    return result;
}

/* Optional: seed function */
static PyObject* fast_seed(PyObject* self, PyObject* args) {
    uint64_t s;
    if (!PyArg_ParseTuple(args, "K", &s))
        return NULL;
    if (s == 0) s = 1;  /* avoid zero state */
    state = s;
    Py_RETURN_NONE;
}

static PyObject* compute_plant_growth_grass(PyObject* self, PyObject* args)
{
    double size, water_relative_height, total_moisture, min_moisture_required, decay_probability, nutrition;

    /* Parse double (float) arguments */
    if (!PyArg_ParseTuple(args, "dddddd", &size, &water_relative_height, &total_moisture,
    &min_moisture_required, &decay_probability, &nutrition)) {
        return NULL;  // TypeError automatically set
    }
    /*
    // implementation of the following python code
        if self.tile.world.water_relative_height[tile_x, tile_y] > 1e-4:  # plant is dying due to high water level
            self.size *= 1 - 0.05 - random.random() * 0.2
        elif total_moisture < self.MIN_MOISTURE_REQUIRED or random.random() < self.RANDOM_DECAY_PROBABILITY:
            self.size *= 1 - 0.01 - random.random() * 0.05 - 0.01 * (1 - self.tile.nutrition)
        else:
            self.size *= 1 + 0.01 + random.random() * 0.1 * self.tile.nutrition
        self.size = min(self.size, 1)
    */
    if (water_relative_height > 0.0001){ // plant is dying due to high water level
        size *= 1 - 0.01 - dFastRandom() * 0.05 - 0.01 * (1 - nutrition);
    }
    else if (total_moisture < min_moisture_required || dFastRandom() < decay_probability){
        size *= 1 - 0.01 - dFastRandom() * 0.05 - 0.01 * (1 - nutrition);
    }
    else {
        size *= 1 + 0.01 + dFastRandom() * 0.1 * nutrition;
    }
    if (size > 1){
    size = 1;
    }
    /* Return Python float */
    return PyFloat_FromDouble(size);
}

/* Method table */
static PyMethodDef methods[] = {
    {"random", fast_random, METH_NOARGS, "Fast random float [0,1)"},
    {"seed", fast_seed, METH_VARARGS, "Seed the RNG"},
    {"compute_plant_growth_grass", compute_plant_growth_grass, METH_VARARGS, "-"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_world_logic",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit__world_logic(void) {
    return PyModule_Create(&module);
}