import torch
import torch.nn as nn

import numpy as np
from cfg.config import cfg
from torch.nn.utils.rnn import pack_padded_sequence
from GLAttention import func_attention


# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def caption_loss(cap_output, captions):
    criterion = nn.CrossEntropyLoss()
    caption_loss = criterion(cap_output, captions)
    return caption_loss


def sent_loss(cnn_code, rnn_code, labels, class_ids, batch_size, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()
    masks = torch.gt(masks, 0)
    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float("inf"))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def words_loss(img_features, words_emb, labels, cap_lens, class_ids, batch_size):
    """
    words_emb(query): batch x nef x seq_len
    img_features(context): batch x nef x 17 x 17
    """
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()
    # masks = torch.gt(masks, 0)

    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float("inf"))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps


# ##################Loss for G and Ds##############################
def discriminator_loss(
    netD,
    paired_real_imgs,
    unpaired_real_imgs,
    paired_fake_imgs,
    paired_sent_emb,
    unpaired_fake_imgs,
    unpaired_sent_emb,
    real_labels,
    fake_labels,
):
    # discriminator loss is calculated for each discriminators
    ## real_labels is a all one tensor
    ## fake_labels is a all zero tensor

    # Forward
    #
    paired_real_features = netD(paired_real_imgs)
    paired_fake_features = netD(paired_fake_imgs.detach())

    unpaired_real_features = netD(unpaired_real_imgs)
    unpaired_fake_features = netD(unpaired_fake_imgs.detach())

    # loss
    ## 3rd term
    paired_cond_real_logits = netD.COND_DNET(paired_real_features, paired_sent_emb)
    paired_cond_real_errD = nn.BCELoss()(paired_cond_real_logits, real_labels)

    # unpaired_cond_real_logits = netD.COND_DNET(
    #     unpaired_real_features, unpaired_sent_emb
    # )
    # unpaired_cond_real_errD = nn.BCELoss()(unpaired_cond_real_logits, real_labels)

    ## 4th term
    paired_cond_fake_logits = netD.COND_DNET(paired_fake_features, paired_sent_emb)
    paired_cond_fake_errD = nn.BCELoss()(paired_cond_fake_logits, fake_labels)

    # unpaired_cond_fake_logits = netD.COND_DNET(
    #     unpaired_fake_features, unpaired_sent_emb
    # )
    # unpaired_cond_fake_errD = nn.BCELoss()(unpaired_cond_fake_logits, fake_labels)

    ##
    paired_batch_size = paired_real_features.size(0)
    paired_cond_wrong_logits = netD.COND_DNET(
        paired_real_features[: (paired_batch_size - 1)],
        paired_sent_emb[1:paired_batch_size],
    )
    paired_cond_wrong_errD = nn.BCELoss()(
        paired_cond_wrong_logits, fake_labels[1:paired_batch_size]
    )
    # unpaired_batch_size = unpaired_real_features.size(0)
    # unpaired_cond_wrong_logits = netD.COND_DNET(
    #     unpaired_real_features[: (unpaired_batch_size - 1)],
    #     unpaired_sent_emb[1:unpaired_batch_size],
    # )
    # unpaired_cond_wrong_errD = nn.BCELoss()(
    #     unpaired_cond_wrong_logits, fake_labels[1:unpaired_batch_size]
    # )

    # netD.UNCOND_DNET is D_GET_LOGITS(ndf, nef, bcondition=False)
    #
    if netD.UNCOND_DNET is not None:
        # real_features --> nn.Conv2d --> nn.Sigmoid
        paired_real_logits = netD.UNCOND_DNET(paired_real_features)
        unpaired_real_logits = netD.UNCOND_DNET(unpaired_real_features)

        # fake_features --> nn.Conv2d --> nn.Sigmoid
        paired_fake_logits = netD.UNCOND_DNET(paired_fake_features)
        unpaired_fake_logits = netD.UNCOND_DNET(unpaired_fake_features)

        paired_real_errD = nn.BCELoss()(paired_real_logits, real_labels)  # 1st term
        paired_fake_errD = nn.BCELoss()(paired_fake_logits, fake_labels)  # 2nd term

        unpaired_real_errD = nn.BCELoss()(unpaired_real_logits, real_labels)  # 1st term
        unpaired_fake_errD = nn.BCELoss()(unpaired_fake_logits, fake_labels)  # 2nd term

        errD = (
            (paired_real_errD + paired_cond_real_errD) / 2.0
            + (paired_fake_errD + paired_cond_fake_errD + paired_cond_wrong_errD) / 3.0
            + (unpaired_real_errD + unpaired_fake_errD) / 2.0
            # + (unpaired_real_errD + unpaired_cond_real_errD) / 2.0
            # + (unpaired_fake_errD + unpaired_cond_fake_errD) / 2.0
        )
    else:
        errD = (
            paired_cond_real_errD
            + (paired_cond_fake_errD + paired_cond_wrong_errD) / 2.0
        )
    return errD


def generator_loss(
    real_labels,
    match_labels,
    netsD,
    image_encoder,
    caption_cnn,
    caption_rnn,
    paired_fake_imgs,
    paired_caps,
    paired_cap_lens,
    paired_class_ids,
    paired_words_embs,
    paired_sent_emb,
    unpaired_fake_imgs,
    unpaired_caps,
    unpaired_cap_lens,
    unpaired_cap_class_ids,
    unpaired_words_embs,
    unpaired_sent_emb,
):
    numDs = len(netsD)
    paired_logs = ""
    unpaired_logs = ""

    # Forward
    #
    errG_total = 0

    for i in range(numDs):
        ################
        # Paired Loss
        ################
        features = netsD[i](paired_fake_imgs[i])  # 1st term

        cond_logits = netsD[i].COND_DNET(features, paired_sent_emb)  # 2nd term
        cond_errG = nn.BCELoss()(cond_logits, real_labels)

        # netsD.UNCOND_DNET are all None
        #
        if netsD[i].UNCOND_DNET is not None:
            logits = netsD[i].UNCOND_DNET(features)
            errG = nn.BCELoss()(logits, real_labels)
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        errG_total += g_loss

        ################
        # Unpaired Loss
        ################
        features = netsD[i](unpaired_fake_imgs[i])  # 1st term

        cond_logits = netsD[i].COND_DNET(features, unpaired_sent_emb)  # 2nd term
        cond_errG = nn.BCELoss()(cond_logits, real_labels)

        # netD.UNCOND_DNET is D_GET_LOGITS(ndf, nef, bcondition=False)
        #
        if netsD[i].UNCOND_DNET is not None:
            logits = netsD[i].UNCOND_DNET(features)
            errG = nn.BCELoss()(logits, real_labels)
            un_g_loss = errG + cond_errG
        else:
            un_g_loss = cond_errG
        errG_total += un_g_loss

        paired_logs += "paired_g_loss%d: %.2f " % (
            i,
            g_loss.data,
        )
        unpaired_logs += "unpaired_g_loss%d: %.2f " % (
            i,
            un_g_loss.data,
        )

        ################
        # STREAM Loss
        ################
        if i == (numDs - 1):
            paired_fakeimg_feature = caption_cnn(paired_fake_imgs[i])
            paired_caps.cuda()
            paired_target_cap = pack_padded_sequence(
                paired_caps, paired_cap_lens.data.tolist(), batch_first=True
            )[0].cuda()
            paired_cap_output = caption_rnn(
                paired_fakeimg_feature, paired_caps, paired_cap_lens
            )
            paired_cap_loss = (
                caption_loss(paired_cap_output, paired_target_cap)
                * cfg.TRAIN.SMOOTH.LAMBDA1
            )

            unpaired_fakeimg_feature = caption_cnn(unpaired_fake_imgs[i])
            unpaired_caps.cuda()
            unpaired_target_cap = pack_padded_sequence(
                unpaired_caps, unpaired_cap_lens.data.tolist(), batch_first=True
            )[0].cuda()
            unpaired_cap_output = caption_rnn(
                unpaired_fakeimg_feature, unpaired_caps, unpaired_cap_lens
            )
            unpaired_cap_loss = (
                caption_loss(unpaired_cap_output, unpaired_target_cap)
                * cfg.TRAIN.SMOOTH.LAMBDA1
            )

            errG_total += paired_cap_loss + unpaired_cap_loss
            paired_logs += "paired_cap_loss: %.2f " % (paired_cap_loss,)
            unpaired_logs += "unpaired_cap_loss: %.2f " % (unpaired_cap_loss,)
    return errG_total, paired_logs, unpaired_logs


##################################################################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD
