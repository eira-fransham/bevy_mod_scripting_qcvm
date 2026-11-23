// Test fixture for `run-qc`. Recompile with `fteqcc -O0` from this directory after editing, and commit
// the resulting `test.dat`.

void() null_function = {};

var float scale = 1;
vector offset;
string greeting;

void(string s) print = #1;

float(float a, float b) add_scaled = {
	return (a + b) * scale;
};

float() offset_sum = {
	return offset_x + offset_y + offset_z;
};

float(float a, float b, float c) sum3 = {
	return a + b + c;
};

float(float x) negate = {
	return -x;
};

string() get_greeting = {
	return greeting;
};

float() call_builtin = {
	print("hello");
	return 1;
};

float(vector v) vector_arg = {
	return v_x;
};

var float(float a, float b) main = add_scaled;
var float(float a, float b) callback;

float(float a, float b) call_callback = {
	return callback(a, b);
};

float(float a, float b) call_main = {
	return main(a, b);
};

float(vector v) vector_sum = {
	return v_x + v_y + v_z;
};

string(string s) string_identity = {
	return s;
};

float(string s) string_is_null = {
	return !s;
};

.float health;
.vector origin;
.string netname;
.entity owner;

float(entity e) entity_is_null = {
	return !e;
};

float(entity e) get_health = {
	return e.health;
};

float(entity e) origin_sum = {
	return e.origin_x + e.origin_y + e.origin_z;
};

string(entity e) get_netname = {
	return e.netname;
};

float(entity e) owner_health = {
	return e.owner.health;
};

float(entity e, float amount) heal = {
	e.health = e.health + amount;
	return e.health;
};

var entity player;

float() player_health = {
	return player.health;
};
