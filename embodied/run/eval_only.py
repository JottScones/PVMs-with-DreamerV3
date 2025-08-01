# eval_only.py

from collections import defaultdict
from functools import partial as bind

import elements
import embodied
import numpy as np


# 1. Update the function signature
def eval_only(make_agent, make_replay_eval, make_stream, make_env, make_logger, args):
    assert args.from_checkpoint

    agent = make_agent()
    # 2. Create the replay buffer and logger
    replay_eval = make_replay_eval()
    logger = make_logger()

    logdir = elements.Path(args.logdir)
    logdir.mkdir()
    print('Logdir', logdir)
    step = logger.step
    usage = elements.Usage(**args.usage)
    agg = elements.Agg()
    epstats = elements.Agg()
    episodes = defaultdict(elements.Agg)

    # 3. Add clocks for logging and reporting
    should_log = elements.when.Clock(args.log_every)
    should_report = elements.when.Clock(args.report_every)  # New
    policy_fps = elements.FPS()

    @elements.timer.section('logfn')
    def logfn(tran, worker):
        episode = episodes[worker]
        tran['is_first'] and episode.reset()
        episode.add('score', tran['reward'], agg='sum')
        episode.add('length', 1, agg='sum')
        episode.add('rewards', tran['reward'], agg='stack')
        for key, value in tran.items():
            isimage = (value.dtype == np.uint8) and (value.ndim == 3)
            if isimage and worker == 0:
                episode.add(f'policy_{key}', value, agg='stack')
            elif not isimage and key.startswith('log/'):
                assert value.ndim == 0, (key, value.shape, value.dtype)
                episode.add(key + '/avg', value, agg='avg')
                episode.add(key + '/max', value, agg='max')
                episode.add(key + '/sum', value, agg='sum')
        if tran['is_last']:
            result = episode.result()
            logger.add({
                'score': result.pop('score'),
                'length': result.pop('length'),
            }, prefix='episode')
            rew = result.pop('rewards')
            if len(rew) > 1:
                result['reward_rate'] = (
                    np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
            epstats.add(result)

    fns = [bind(make_env, i) for i in range(args.envs)]
    driver = embodied.Driver(fns, parallel=(not args.debug))
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(lambda tran, _: policy_fps.step())
    # 4. Add data to the replay buffer on each environment step
    driver.on_step(replay_eval.add)
    driver.on_step(logfn)

    # 5. Create the data stream and initialize the report carry
    stream_eval = iter(agent.stream(make_stream(replay_eval, 'eval')))
    carry_eval = agent.init_report(args.batch_size)

    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(args.from_checkpoint, keys=['agent'])

    print('Start evaluation')
    policy = lambda *args: agent.policy(*args, mode='eval')
    driver.reset(agent.init_policy)
    while step < args.steps:
        driver(policy, steps=10)

        # 6. Add the reporting loop
        if should_report(step) and len(replay_eval) >= args.batch_size * args.report_length:
            print("Generating report with loss metrics...")
            # The agent.report call computes the losses
            print(stream_eval)
            carry_eval, mets = agent.report(carry_eval, next(stream_eval))
            logger.add(mets, prefix='eval')
            print("Report generated.")
            replay_eval.save()

        if should_log(step):
            logger.add(agg.result())
            logger.add(epstats.result(), prefix='epstats')
            # Add replay stats to the log
            logger.add(replay_eval.stats(), prefix='replay_eval')
            logger.add(usage.stats(), prefix='usage')
            logger.add({'fps/policy': policy_fps.result()})
            logger.add({'timer': elements.timer.stats()['summary']})
            logger.write()

    logger.close()
