package org.mystic.actors

import akka.actor.{Actor, ActorLogging}
import akka.io.Tcp.{PeerClosed, Received}
import org.mystic.model.Deal

class ReadHandler extends Actor with ActorLogging {

  override def receive: Receive = {

    case Received(data) =>
      val storage = context.system.actorSelection("user/storage")
      storage ! Deal.byteString2Deal(data)

    case PeerClosed =>
      log.info(s"peer closed")
      context stop self
  }
}
