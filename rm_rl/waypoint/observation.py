"""Retained conservative radar input contract; reject incompatible experiments."""
from rm_rl.waypoint import radar, protocol27
SCHEMAS=(radar.SCHEMA, protocol27.SCHEMA)


def description(schema):
    if schema == protocol27.SCHEMA:
        return protocol27.FEATURE_DIM, protocol27.contract()
    if schema!=radar.SCHEMA:
        raise ValueError(f'unsupported radar observation schema: {schema}')
    return radar.FEATURE_DIM,radar.contract()


def adapter(schema,config,team,seed):
    description(schema)
    if schema == protocol27.SCHEMA:
        from .protocol27_sim import ProtocolAdapter
        return ProtocolAdapter(config,team,seed)
    return radar.RadarAdapter(config,team,seed)


def reader(schema):
    description(schema)
    return protocol27.value if schema == protocol27.SCHEMA else radar.value


def field_index(schema, name):
    description(schema)
    return (protocol27.FIELD_INDEX if schema == protocol27.SCHEMA else radar.FIELD_INDEX)[name]
