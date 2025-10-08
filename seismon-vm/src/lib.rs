use std::{borrow::Cow, marker::PhantomData, sync::Arc};

use bevy_ecs::{
    entity::Entity,
    system::{In, IntoSystem, SystemId},
    world::World,
};
use bevy_reflect::{DynamicTuple, Tuple, Typed};

use crate::value::{IntoValue, Value};

mod map;
pub mod value;

pub type Getter<Entity> = Arc<dyn Fn(Entity) -> Value>;

pub struct Vm<Ctx, Entity> {
    getters: Vec<(Cow<'static, str>, Getter<Entity>)>,
    _ctx: PhantomData<Ctx>,
}

impl Vm<Ctx, Entity> {
    pub fn field<F>(&mut self, name: Arc<F>, func_sys: F) -> &mut Self
    where
        Arc<F>: CoerceUnsized<Getter<Entity>>,
    {
        let sys = self
            .world
            .register_system(func_sys.pipe(|In(val): In<O>| val.into_seismon_value()));

        self.getters.push((name.into(), sys));

        self
    }

    pub fn builtin<F, I, O, M>(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        func_sys: F,
    ) -> &mut Self
    where
        I: Typed + Tuple,
        O: IntoValue,
        F: IntoSystem<In<I>, O, M> + 'static,
        O: IntoValue,
    {
        let convert_args = |In(scalars): In<&[Scalar]>| -> I {
            let mut args: DynamicTuple = DynamicTuple::default();

            todo!()
        };
        let sys = self.world.register_system(convert_args.pipe(func_sys));
        todo!();
        self
    }
}
