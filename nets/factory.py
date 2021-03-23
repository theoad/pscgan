import nets.encdec as encdec
# import nets.stylegan2 as sg2
import nets.dncnn as dncnn
import nets.critic as critic

factory = {
    'encdec': encdec.EncDec,
    'sg2_disc': critic.BasicCritic,
    # 'sg2_disc_lag': sg2.DiscriminatorLAG,
    'dncnn': dncnn.DnCNN,
    # 'ydisc': sg2.YDiscriminator,
    'y_preprocess': critic.YPreProcessCritic,
    'y_postprocess': critic.YPostProcessCritic,
    'y_preprocess_old': critic.YPreProcessCriticOld,
    'y_preprocess_old_lag': critic.YPreProcessCriticOldLAG
}